from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime.registry import ProviderRegistry


class RuntimeContractTest(unittest.TestCase):
    def test_reports_the_local_runtime_without_exposing_secrets(self) -> None:
        client = TestClient(create_app(registry=ProviderRegistry(), require_api_key=False))

        response = client.get("/runtime")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("omni-providers", payload["service"])
        self.assertEqual("127.0.0.1", payload["host"])
        self.assertEqual(7814, payload["port"])
        self.assertNotIn("apiKey", payload)


if __name__ == "__main__":
    unittest.main()
