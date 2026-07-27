from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nhs_rtt_pipeline.rtt_parsing import (
    coalesce_numeric,
    extract_publication_month_from_path,
    financial_year_slug,
    parse_month_from_current_period,
    parse_month_from_legacy_period,
    parse_period_from_text,
    standardise_part,
)


class RttParsingTests(unittest.TestCase):
    def test_date_extraction_supports_october_2015_start(self) -> None:
        self.assertEqual(parse_period_from_text("RTT Oct 2015 Full CSVs"), pd.Timestamp("2015-10-01"))
        self.assertEqual(extract_publication_month_from_path("rtt-waiting-times-October-2015.zip"), pd.Timestamp("2015-10-01"))
        self.assertEqual(parse_month_from_current_period("31/10/2015"), pd.Timestamp("2015-10-01"))
        self.assertEqual(parse_month_from_legacy_period("October", 2015), pd.Timestamp("2015-10-01"))

    def test_financial_year_and_part_standardisation(self) -> None:
        self.assertEqual(financial_year_slug(2015), "2015-16")
        self.assertEqual(standardise_part("Part 2A incomplete pathways"), "PART_2A")
        self.assertEqual(standardise_part("part_1b admitted adjusted"), "PART_1B")

    def test_coalesce_numeric_strips_common_publication_formatting(self) -> None:
        frame = pd.DataFrame({"a": [None, "1,200", "*"], "b": ["3", "4", "5"]})
        values = coalesce_numeric(frame, ["a", "b"])

        self.assertEqual(values.tolist(), [3.0, 1200.0, 5.0])


if __name__ == "__main__":
    unittest.main()
