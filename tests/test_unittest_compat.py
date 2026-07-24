"""Run the same plain assertion tests when pytest is unavailable locally."""

from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipIf("pytest" in sys.modules, "pytest collects the underlying assertion tests directly")
class PhaseOneContractTests(unittest.TestCase):
    def test_pytest_discoverable_contract_suite(self) -> None:
        for filename in (
            "test_public_api.py", "test_contracts.py", "test_config.py", "test_submission.py",
            "test_cli.py", "test_project_integration.py",
        ):
            namespace = runpy.run_path(str(ROOT / "tests" / filename))
            for name, function in namespace.items():
                if name.startswith("test_") and callable(function):
                    with self.subTest(filename=filename, test=name):
                        function()
