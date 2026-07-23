from __future__ import annotations

import uvicorn

from app.config import HOST, PORT
from app.runtime.playwright_runtime import configure_external_playwright_node

configure_external_playwright_node()

from app.main import app as application


def main() -> None:
    uvicorn.run(application, host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
