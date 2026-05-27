from __future__ import annotations

import unittest

import pandas as pd

from project_time.cleaning import CleanConfig, clean_timeseries, infer_columns


class CleaningTests(unittest.TestCase):
    def test_infers_columns(self) -> None:
        data = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Well ID": ["A", "A"],
                "Water Level": [1.0, 2.0],
            }
        )
        inferred = infer_columns(data)
        self.assertEqual(inferred["date_column"], "Date")
        self.assertEqual(inferred["value_column"], "Water Level")
        self.assertEqual(inferred["id_column"], "Well ID")

    def test_cleans_duplicates_and_interpolates(self) -> None:
        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-03"],
                "site": ["A", "A", "A"],
                "level": [1.0, 3.0, 5.0],
            }
        )
        result = clean_timeseries(
            data,
            CleanConfig(
                date_column="date",
                value_column="level",
                id_column="site",
                frequency="D",
            ),
        )
        self.assertEqual(result.report["duplicates_collapsed"], 1)
        self.assertEqual(len(result.data), 3)
        self.assertEqual(result.data.loc[0, "value"], 2.0)
        self.assertEqual(result.data.loc[1, "value"], 3.5)


if __name__ == "__main__":
    unittest.main()
