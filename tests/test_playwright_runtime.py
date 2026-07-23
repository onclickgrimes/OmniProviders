from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.runtime.playwright_runtime import (
    configure_external_playwright_node,
    resolve_playwright_node_path,
)


class PlaywrightRuntimeTest(unittest.TestCase):
    def test_uses_inherited_external_electron_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            electron = Path(directory) / "electron.exe"
            electron.touch()
            with patch.dict(
                os.environ,
                {"PLAYWRIGHT_NODEJS_PATH": str(electron)},
                clear=False,
            ):
                self.assertEqual(str(electron), resolve_playwright_node_path())
                self.assertEqual(str(electron), configure_external_playwright_node())
                self.assertEqual("1", os.environ["ELECTRON_RUN_AS_NODE"])

    def test_explicit_driver_has_priority_over_inherited_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit-node.exe"
            inherited = Path(directory) / "inherited-node.exe"
            explicit.touch()
            inherited.touch()
            with patch.dict(
                os.environ,
                {
                    "OMNIPROVIDERS_PLAYWRIGHT_NODE_PATH": str(explicit),
                    "PLAYWRIGHT_NODEJS_PATH": str(inherited),
                },
                clear=False,
            ):
                self.assertEqual(str(explicit), resolve_playwright_node_path())


if __name__ == "__main__":
    unittest.main()
