from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from app import config
from app.persistence.credentials import get_env_or_credential
from app.runtime.playwright_runtime import (
    configure_external_playwright_node,
    resolve_playwright_node_path,
)


def data_dir() -> Path:
    return config.DATA_DIR


FLOW_SITE_KEY = "6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV"
FLOW_HOME_URL = "https://labs.google/fx/tools/flow"

configure_external_playwright_node()

try:
    from playwright.async_api import BrowserContext, Page, async_playwright
except Exception:  # pragma: no cover - runtime dependency validation.
    BrowserContext = None  # type: ignore[assignment]
    Page = None  # type: ignore[assignment]
    async_playwright = None  # type: ignore[assignment]


class FlowBrowserCaptchaError(RuntimeError):
    pass


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


class FlowBrowserCaptchaService:
    def __init__(self) -> None:
        self.profile_dir = data_dir() / "flow-browser-profile"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright: Any | None = None
        self._context: BrowserContext | None = None
        self._context_headless: bool | None = None
        self._login_page: Page | None = None
        self._lock = asyncio.Lock()
        self._solve_semaphore: asyncio.Semaphore | None = None
        self._last_token_at: float | None = None
        self._last_fingerprint: dict[str, Any] | None = None
        self._successful_solves_since_context_start = 0
        self._fresh_restart_pending = False
        self._pending_pages: list[Page] = []
        self._pending_pages_lock = asyncio.Lock()

    def _available(self) -> bool:
        return async_playwright is not None

    def _headless(self, override: bool | None = None) -> bool:
        if override is not None:
            return override
        return _truthy(
            get_env_or_credential("FLOW_BROWSER_HEADLESS", "PERSONAL_BROWSER_HEADLESS"),
            default=True,
        )

    def _foreground_captcha(self) -> bool:
        return _truthy(
            get_env_or_credential("FLOW_BROWSER_FOREGROUND", "FLOW_BROWSER_BRING_TO_FRONT"),
            default=False,
        )

    def _timeout_ms(self) -> int:
        value = get_env_or_credential("FLOW_BROWSER_TIMEOUT") or "60"
        try:
            return max(10, int(float(value))) * 1000
        except ValueError:
            return 60000

    def _settle_seconds(self) -> float:
        value = get_env_or_credential("FLOW_BROWSER_RECAPTCHA_SETTLE_SECONDS") or "3"
        try:
            return max(0.0, float(value))
        except ValueError:
            return 3.0

    def _warmup_seconds(self) -> float:
        value = get_env_or_credential("FLOW_BROWSER_WARMUP_SECONDS") or "4"
        try:
            return max(0.0, float(value))
        except ValueError:
            return 4.0

    def _concurrency(self) -> int:
        value = get_env_or_credential("FLOW_BROWSER_COUNT", "FLOW_BROWSER_CAPTCHA_CONCURRENCY") or "1"
        try:
            return max(1, int(value))
        except ValueError:
            return 1

    def _proxy(self) -> dict[str, str] | None:
        proxy_url = self._proxy_url()
        if not proxy_url:
            return None
        return {"server": proxy_url}

    def _proxy_url(self) -> str | None:
        return get_env_or_credential("FLOW_BROWSER_PROXY", "FLOW_PROXY") or None

    def _page_hold_seconds(self) -> float:
        value = get_env_or_credential("FLOW_BROWSER_PAGE_HOLD_SECONDS") or "20"
        try:
            return max(0.0, float(value))
        except ValueError:
            return 20.0

    def _fresh_restart_every_n_solves(self) -> int:
        value = get_env_or_credential(
            "FLOW_BROWSER_FRESH_RESTART_EVERY_N_SOLVES",
            "BROWSER_PERSONAL_FRESH_RESTART_EVERY_N_SOLVES",
        ) or "10"
        try:
            return max(0, min(1000, int(value)))
        except ValueError:
            return 10

    def _browser_channel(
        self,
        *,
        headless: bool | None = None,
        foreground: bool = False,
    ) -> str | None:
        value = get_env_or_credential("FLOW_BROWSER_CHANNEL")
        if value is None:
            return "chrome"
        normalized = value.strip()
        if not normalized or normalized.lower() in {"none", "chromium", "playwright"}:
            return None
        return normalized

    def _browser_executable(
        self,
        *,
        headless: bool | None = None,
        foreground: bool = False,
    ) -> str | None:
        value = get_env_or_credential("FLOW_BROWSER_EXECUTABLE")
        if not value:
            candidates: list[str] = []
            channel = (self._browser_channel(headless=headless, foreground=foreground) or "").lower()
            if channel == "chrome":
                candidates = [
                    os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(os.environ.get("LocalAppData", ""), "Google", "Chrome", "Application", "chrome.exe"),
                ]
            elif channel == "msedge":
                candidates = [
                    os.path.join(os.environ.get("ProgramFiles", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
                    os.path.join(os.environ.get("LocalAppData", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
                ]
            for candidate in candidates:
                path = Path(candidate)
                if path.exists():
                    return str(path)
            return None
        path = Path(value.strip())
        return str(path) if path.exists() else None

    def _flow_url(self, project_id: str | None = None) -> str:
        project = (project_id or "").strip()
        return f"{FLOW_HOME_URL}/project/{project}" if project else FLOW_HOME_URL

    def _is_closed_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            fragment in message
            for fragment in (
                "target page, context or browser has been closed",
                "target closed",
                "browser has been closed",
                "browser closed",
                "context has been closed",
                "context closed",
                "connection closed",
            )
        )

    def _mark_context_closed(self, context: BrowserContext | None = None) -> None:
        if context is not None and self._context is not context:
            return
        self._context = None
        self._context_headless = None
        self._login_page = None
        self._solve_semaphore = None
        self._pending_pages = []
        self._successful_solves_since_context_start = 0
        self._fresh_restart_pending = False

    async def _reset_context(self, expected_context: BrowserContext | None = None) -> None:
        async with self._lock:
            if expected_context is not None and self._context is not expected_context:
                return
            context = self._context
            playwright = self._playwright
            self._mark_context_closed(context)
            self._playwright = None
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if playwright:
                try:
                    await playwright.stop()
                except Exception:
                    pass

    async def _close_context_locked(self) -> None:
        context = self._context
        playwright = self._playwright
        self._mark_context_closed(context)
        self._playwright = None
        if context:
            try:
                await context.close()
            except Exception:
                pass
        if playwright:
            try:
                await playwright.stop()
            except Exception:
                pass

    async def _ensure_context(
        self,
        *,
        headless: bool | None = None,
        foreground: bool = False,
    ) -> BrowserContext:
        if not self._available():
            raise FlowBrowserCaptchaError(
                "Playwright is not available in OmniProviders. Install the Python Playwright package; "
                "the browser uses installed Chrome with the external Electron/Node driver."
            )

        desired_headless = self._headless(headless)
        async with self._lock:
            if self._context:
                if self._context_headless == desired_headless:
                    return self._context
                await self._close_context_locked()

            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

            self._playwright = await async_playwright().start()
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-first-run",
            ]
            if not foreground and not self._foreground_captcha():
                launch_args.append("--start-minimized")
                launch_args.append("--window-position=-32000,-32000")
                launch_args.append("--window-size=1,1")
            launch_options: dict[str, Any] = {
                "headless": desired_headless,
                "viewport": {"width": 1366, "height": 820},
                "locale": "pt-BR",
                "args": launch_args,
            }
            proxy = self._proxy()
            if proxy:
                launch_options["proxy"] = proxy

            executable = self._browser_executable(headless=desired_headless, foreground=foreground)
            channel = self._browser_channel(headless=desired_headless, foreground=foreground)
            if executable:
                launch_options["executable_path"] = executable
            elif channel:
                launch_options["channel"] = channel

            try:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.profile_dir),
                    **launch_options,
                )
            except Exception as configured_browser_error:
                if executable or channel:
                    launch_options.pop("executable_path", None)
                    launch_options.pop("channel", None)
                    try:
                        self._context = await self._playwright.chromium.launch_persistent_context(
                            user_data_dir=str(self.profile_dir),
                            **launch_options,
                        )
                    except Exception as fallback_error:
                        configured_browser = executable or channel
                        raise FlowBrowserCaptchaError(
                            f"Could not launch configured browser '{configured_browser}': "
                            f"{configured_browser_error}. Playwright Chromium fallback also failed: "
                            f"{fallback_error}"
                        ) from fallback_error
                else:
                    raise
            self._context.set_default_timeout(self._timeout_ms())
            current_context = self._context
            self._context_headless = desired_headless
            current_context.on("close", lambda *_: self._mark_context_closed(current_context))
            self._solve_semaphore = asyncio.Semaphore(self._concurrency())
            self._successful_solves_since_context_start = 0
            self._fresh_restart_pending = False
            return self._context

    async def _new_page(self, project_id: str | None = None) -> Page:
        context = await self._ensure_context()
        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        await page.goto(self._flow_url(project_id), wait_until="domcontentloaded", timeout=self._timeout_ms())
        return page

    async def _warm_page(self, page: Page) -> None:
        try:
            foreground = self._foreground_captcha()
            if foreground:
                await page.bring_to_front()
            await page.mouse.move(320, 220)
            await page.mouse.move(540, 320, steps=10)
            await page.mouse.wheel(0, 240)
            await page.evaluate(
                """
                (foreground) => {
                  try {
                    if (foreground) window.focus();
                    window.dispatchEvent(new Event('focus'));
                    document.dispatchEvent(new MouseEvent('mousemove', {
                      bubbles: true,
                      clientX: Math.max(32, Math.floor((window.innerWidth || 1280) * 0.42)),
                      clientY: Math.max(32, Math.floor((window.innerHeight || 720) * 0.36))
                    }));
                  } catch (error) {}
                }
                """,
                foreground,
            )
        except Exception:
            pass

        warmup = self._warmup_seconds()
        if warmup > 0:
            await asyncio.sleep(warmup)

    async def _ensure_recaptcha(self, page: Page) -> None:
        await page.evaluate(
            """
            (siteKey) => {
              const hasScript = Array.from(document.scripts || []).some((script) => {
                const src = script && script.src ? script.src : '';
                return src.includes('/recaptcha/enterprise.js');
              });
              if (hasScript) return;
              const script = document.createElement('script');
              script.src = `https://www.google.com/recaptcha/enterprise.js?render=${siteKey}`;
              script.async = true;
              document.head.appendChild(script);
            }
            """,
            FLOW_SITE_KEY,
        )
        await page.wait_for_function(
            """
            () => typeof grecaptcha !== 'undefined'
              && typeof grecaptcha.enterprise !== 'undefined'
              && typeof grecaptcha.enterprise.execute === 'function'
            """,
            timeout=self._timeout_ms(),
        )

    async def _execute_recaptcha(self, page: Page, action: str) -> str:
        token = await page.evaluate(
            """
            ({siteKey, action}) => new Promise((resolve, reject) => {
              const timeout = setTimeout(() => reject(new Error('reCAPTCHA execute timeout')), 30000);
              try {
                grecaptcha.enterprise.ready(() => {
                  grecaptcha.enterprise.execute(siteKey, {action})
                    .then((token) => {
                      clearTimeout(timeout);
                      resolve(token);
                    })
                    .catch((error) => {
                      clearTimeout(timeout);
                      reject(error);
                    });
                });
              } catch (error) {
                clearTimeout(timeout);
                reject(error);
              }
            })
            """,
            {"siteKey": FLOW_SITE_KEY, "action": action},
        )
        token = str(token or "").strip()
        if not token:
            raise FlowBrowserCaptchaError("Browser did not return a reCAPTCHA token.")
        settle = self._settle_seconds()
        if settle > 0:
            await asyncio.sleep(settle)
        self._last_token_at = time.time()
        self._successful_solves_since_context_start += 1
        restart_every = self._fresh_restart_every_n_solves()
        if restart_every > 0 and self._successful_solves_since_context_start >= restart_every:
            self._fresh_restart_pending = True
        return token

    async def _capture_page_fingerprint(self, page: Page) -> dict[str, Any] | None:
        try:
            fingerprint = await page.evaluate(
                """
                () => {
                  const ua = navigator.userAgent || "";
                  const lang = navigator.language || "";
                  const uaData = navigator.userAgentData || null;
                  let secChUa = "";
                  let secChUaMobile = "";
                  let secChUaPlatform = "";

                  if (uaData) {
                    if (Array.isArray(uaData.brands) && uaData.brands.length > 0) {
                      secChUa = uaData.brands
                        .map((item) => `"${item.brand}";v="${item.version}"`)
                        .join(", ");
                    }
                    secChUaMobile = uaData.mobile ? "?1" : "?0";
                    if (uaData.platform) secChUaPlatform = `"${uaData.platform}"`;
                  }

                  return {
                    user_agent: ua,
                    accept_language: lang,
                    sec_ch_ua: secChUa,
                    sec_ch_ua_mobile: secChUaMobile,
                    sec_ch_ua_platform: secChUaPlatform,
                  };
                }
                """
            )
            if not isinstance(fingerprint, dict):
                return None

            result: dict[str, Any] = {"proxy_url": self._proxy_url() or ""}
            for key in (
                "user_agent",
                "accept_language",
                "sec_ch_ua",
                "sec_ch_ua_mobile",
                "sec_ch_ua_platform",
            ):
                value = fingerprint.get(key)
                if isinstance(value, str) and value:
                    result[key] = value
            self._last_fingerprint = result
            return result
        except Exception:
            return None

    def get_last_fingerprint(self) -> dict[str, Any] | None:
        return dict(self._last_fingerprint) if self._last_fingerprint else None

    async def _close_page_quietly(self, page: Page | None) -> None:
        if not page:
            return
        try:
            if not page.is_closed():
                await page.close()
        except Exception:
            pass

    async def _close_page_after_timeout(self, page: Page, delay_seconds: float) -> None:
        await asyncio.sleep(delay_seconds)
        should_close = False
        async with self._pending_pages_lock:
            if page in self._pending_pages:
                self._pending_pages.remove(page)
                should_close = True
        if should_close:
            await self._close_page_quietly(page)

    async def _hold_page_for_request(self, page: Page) -> None:
        hold_seconds = self._page_hold_seconds()
        if hold_seconds <= 0:
            await self._close_page_quietly(page)
            return

        pages_to_close: list[Page] = []
        async with self._pending_pages_lock:
            self._pending_pages.append(page)
            max_pending = max(2, self._concurrency() * 2 + 2)
            while len(self._pending_pages) > max_pending:
                pages_to_close.append(self._pending_pages.pop(0))

        for old_page in pages_to_close:
            await self._close_page_quietly(old_page)

        asyncio.create_task(self._close_page_after_timeout(page, hold_seconds))

    async def report_request_finished(self) -> None:
        page: Page | None = None
        pending_count = 0
        async with self._pending_pages_lock:
            if self._pending_pages:
                page = self._pending_pages.pop(0)
            pending_count = len(self._pending_pages)
        await self._close_page_quietly(page)
        if self._fresh_restart_pending and pending_count == 0:
            await self._reset_context()

    async def report_error(self, error_reason: str | None = None) -> None:
        reason = (error_reason or "").lower()
        if "recaptcha" in reason and ("failed" in reason or "unusual" in reason or "403" in reason):
            await self._reset_context()

    async def get_token(
        self,
        project_id: str,
        action: str = "IMAGE_GENERATION",
        *,
        headless: bool | None = None,
    ) -> str:
        headless_override = _optional_bool(headless)
        last_error: Exception | None = None
        for attempt in range(2):
            context: BrowserContext | None = None
            page: Page | None = None
            try:
                context = await self._ensure_context(headless=headless_override)
                if self._solve_semaphore is None:
                    self._solve_semaphore = asyncio.Semaphore(self._concurrency())

                async with self._solve_semaphore:
                    page = await context.new_page()
                    await page.add_init_script(
                        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                    )
                    await page.goto(
                        self._flow_url(project_id),
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms(),
                    )
                    await self._warm_page(page)
                    await self._ensure_recaptcha(page)
                    await self._capture_page_fingerprint(page)
                    token = await self._execute_recaptcha(page, action)
                    await self._hold_page_for_request(page)
                    page = None
                    return token
            except Exception as exc:
                last_error = exc
                if attempt == 0 and self._is_closed_error(exc):
                    await self._reset_context(context)
                    continue
                raise FlowBrowserCaptchaError(f"Flow browser reCAPTCHA failed: {exc}") from exc
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass

        raise FlowBrowserCaptchaError(f"Flow browser reCAPTCHA failed: {last_error}")

    async def open_login_window(self) -> dict[str, Any]:
        for attempt in range(2):
            context: BrowserContext | None = None
            try:
                context = await self._ensure_context(headless=False, foreground=True)
                if self._login_page and not self._login_page.is_closed():
                    await self._login_page.bring_to_front()
                    return {"success": True, "isOpen": True, "loginUrl": FLOW_HOME_URL}

                self._login_page = await context.new_page()
                await self._login_page.goto(FLOW_HOME_URL, wait_until="domcontentloaded", timeout=self._timeout_ms())
                await self._login_page.bring_to_front()
                return {"success": True, "isOpen": True, "loginUrl": FLOW_HOME_URL}
            except Exception as exc:
                if attempt == 0 and self._is_closed_error(exc):
                    await self._reset_context(context)
                    continue
                raise FlowBrowserCaptchaError(f"Flow browser login failed: {exc}") from exc
        raise FlowBrowserCaptchaError("Flow browser login failed.")

    async def refresh_session_token(self, project_id: str | None = None) -> str | None:
        for attempt in range(2):
            context: BrowserContext | None = None
            page: Page | None = None
            close_page = False
            try:
                context = await self._ensure_context(headless=False)
                page = self._login_page if self._login_page and not self._login_page.is_closed() else None
                if page is None:
                    page = await context.new_page()
                    close_page = True
                await page.goto(self._flow_url(project_id), wait_until="domcontentloaded", timeout=self._timeout_ms())
                await asyncio.sleep(2)
                cookies = await context.cookies(["https://labs.google", "https://accounts.google.com"])
                for cookie in cookies:
                    if cookie.get("name") == "__Secure-next-auth.session-token" and cookie.get("value"):
                        return str(cookie["value"])
                return None
            except Exception as exc:
                if attempt == 0 and self._is_closed_error(exc):
                    await self._reset_context(context)
                    continue
                raise FlowBrowserCaptchaError(f"Flow browser session refresh failed: {exc}") from exc
            finally:
                if close_page:
                    try:
                        await page.close()
                    except Exception:
                        pass
        return None

    def status(self) -> dict[str, Any]:
        driver = resolve_playwright_node_path()
        browser_executable = self._browser_executable(
            headless=self._context_headless if self._context is not None else self._headless()
        )
        return {
            "available": self._available() and bool(driver),
            "isOpen": self._context is not None,
            "profileDir": str(self.profile_dir),
            "driver": "external_node",
            "driverExecutable": driver,
            "bundledBrowser": False,
            "headless": self._context_headless if self._context is not None else self._headless(),
            "configuredHeadless": self._headless(),
            "foregroundCaptcha": self._foreground_captcha(),
            "browserChannel": self._browser_channel(
                headless=self._context_headless if self._context is not None else self._headless()
            ) or "playwright-chromium",
            "browserExecutable": browser_executable,
            "lastTokenAt": self._last_token_at,
            "browserCount": self._concurrency(),
            "hasLastFingerprint": bool(self._last_fingerprint),
            "successfulSolvesSinceContextStart": self._successful_solves_since_context_start,
            "freshRestartPending": self._fresh_restart_pending,
        }

    async def close(self) -> None:
        async with self._lock:
            context = self._context
            playwright = self._playwright
            pending_pages = self._pending_pages
            self._context = None
            self._login_page = None
            self._solve_semaphore = None
            self._playwright = None
            self._pending_pages = []
            self._last_fingerprint = None
            self._successful_solves_since_context_start = 0
            self._fresh_restart_pending = False
            for page in pending_pages:
                try:
                    await page.close()
                except Exception:
                    pass
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if playwright:
                try:
                    await playwright.stop()
                except Exception:
                    pass


flow_browser_captcha_service = FlowBrowserCaptchaService()
