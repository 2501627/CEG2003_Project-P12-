import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import polars as pl

from dashboard.monthly_alerts import display_alert_summary


class DashboardMonthlyAlertsTest(unittest.TestCase):
    def test_older_snapshot_counts_the_entire_month(self):
        weights = pd.DataFrame({
            "alert_type": ["Braking", "Fatigue"],
            "weight": [0.25, 0.75],
            "training_records": [900, 100],
        })
        snapshot = SimpleNamespace(report={"alert_week_start": "2024-03-18"}, weights=weights)
        alerts = pl.DataFrame({
            "alert_start": [datetime(2024, 2, 29), datetime(2024, 3, 1),
                            datetime(2024, 3, 31, 23, 59), datetime(2024, 4, 1)],
            "alert_type": ["Fatigue", "Braking", "Braking", "Fatigue"],
        })
        with TemporaryDirectory() as folder:
            path = Path(folder) / "alerts.parquet"
            alerts.write_parquet(path)
            summary = display_alert_summary(snapshot, path)

        self.assertEqual(summary.alert_type.tolist(), ["Braking", "Fatigue"])
        self.assertEqual(summary.month_records.tolist(), [2, 0])
        self.assertEqual(summary.weight.tolist(), [0.25, 0.75])

    def test_current_snapshot_reuses_its_monthly_counts(self):
        summary = pd.DataFrame({"alert_type": ["Braking"], "month_records": [2], "weight": [0.25]})
        snapshot = SimpleNamespace(alert_summary=summary)
        self.assertIs(display_alert_summary(snapshot, None), summary)
