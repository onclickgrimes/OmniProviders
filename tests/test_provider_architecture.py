from __future__ import annotations

import ast
import unittest
from pathlib import Path


DOMAIN_IMPORT_PREFIXES = (
    "app.services",
    "core.app",
)


class ProviderArchitectureTest(unittest.TestCase):
    def test_provider_adapters_do_not_import_video_studio_or_other_providers(self) -> None:
        provider_root = Path(__file__).resolve().parents[1] / "app" / "providers"
        offenders: list[str] = []
        for path in provider_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            if any(imported.startswith(DOMAIN_IMPORT_PREFIXES) for imported in imports):
                offenders.append(str(path.relative_to(provider_root)))
        self.assertEqual([], offenders)

    def test_browser_automation_is_isolated_to_flow_provider(self) -> None:
        provider_root = Path(__file__).resolve().parents[1] / "app" / "providers"
        imports = [
            path.relative_to(provider_root).as_posix()
            for path in provider_root.rglob("*.py")
            if "playwright" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(["flow/browser_captcha.py"], imports)
        code = "\n".join(
            path.read_text(encoding="utf-8") for path in provider_root.rglob("*.py")
        )
        self.assertNotIn("import selenium", code)


if __name__ == "__main__":
    unittest.main()
