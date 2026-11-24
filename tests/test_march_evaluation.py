import unittest
from datetime import date, datetime

import pandas as pd
import polars as pl

from training import weekly_ann as base
from training.weighted_weekly_ann import final_ratings


class MarchEvaluationTest(unittest.TestCase):
    def test_official_metrics_weight_folds_equally(self):
        # The smaller, incorrect fold must contribute one fifth to both metrics.
        examples = [(1, 0, 0.8), (1, 1, 0.2)]
        for fold in range(2, 6):
            examples.extend([(fold, 0, 0.1), (fold, 0, 0.2), (fold, 1, 0.9)])
        rows = pd.DataFrame(examples, columns=["evaluation_fold", base.TARGET, "review_score"])
        rows["captain_id"] = range(len(rows))
        self.assertEqual(final_ratings(rows), {"roc_auc": 0.8, "accuracy": 0.8})
        with self.assertRaises(ValueError):
            final_ratings(pd.concat([rows, rows.iloc[:1]]))

    def test_alerts_end_before_the_first_outcome_day(self):
        alerts = pl.DataFrame({
            "captain_id": [1, 1, 2, 2, 3, 4],
            "alert_start": [datetime(2024, 3, day) for day in [18, 25, 17, 24, 20, 21]],
            "alert_type": ["Harsh Braking"] * 6,
        }).lazy()
        # March 25 and 31 belong to the outcome week. March 24 and April 1 fall outside it.
        reports = pl.DataFrame({
            "captain_id": [1, 2, 3, 4],
            "accident_date": [date(2024, 3, 25), date(2024, 3, 24), date(2024, 3, 31), date(2024, 4, 1)],
        })
        profiles = base.weekly_profiles(alerts, reports, base.FINAL_START)
        self.assertEqual(profiles.alert_rows.tolist(), [1, 1, 1, 1])
        self.assertEqual(profiles[base.TARGET].tolist(), [1, 0, 1, 0])
