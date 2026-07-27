from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.startup_validation import build_startup_status


class DashboardStartupValidationTests(unittest.TestCase):
    def test_startup_status_marks_required_missing_files(self) -> None:
        required = {
            "missing_required": PROJECT_ROOT / "does_not_exist.parquet",
            "present_required": PROJECT_ROOT / "README.md",
        }
        optional = {
            "missing_optional": PROJECT_ROOT / "optional_missing.csv",
        }

        status = build_startup_status(required_files=required, optional_files=optional)

        missing_required = status[status["required"] & ~status["exists"]]
        self.assertEqual(set(missing_required["artifact"]), {"missing_required"})
        self.assertIn("present_required", set(status.loc[status["exists"], "artifact"]))
        self.assertIn("missing_optional", set(status["artifact"]))


if __name__ == "__main__":
    unittest.main()
