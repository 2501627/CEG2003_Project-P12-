import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from dashboard.roc import roc_points


class DashboardRocTest(unittest.TestCase):
    def test_mean_preserves_vertical_jumps_and_tied_scores(self):
        predictions = pd.DataFrame({
            "evaluation_fold": [1, 1, 2, 2, 2, 2],
            "reported_next_week": [1, 0, 0, 1, 0, 1],
            "review_score": [.9, .1, .8, .5, .5, .1],
        })
        aucs = []
        for fold, part in predictions.groupby("evaluation_fold"):
            expected_x, expected_y, _ = roc_curve(part.reported_next_week, part.review_score)
            curve = roc_points(predictions, fold)
            np.testing.assert_array_equal(curve.false_positive_rate, expected_x)
            np.testing.assert_array_equal(curve.detection_rate, expected_y)
            aucs.append(roc_auc_score(part.reported_next_week, part.review_score))

        mean = roc_points(predictions)
        self.assertAlmostEqual(np.trapezoid(mean.detection_rate, mean.false_positive_rate), np.mean(aucs))
        self.assertEqual(mean.loc[mean.false_positive_rate == 0, "detection_rate"].tolist(), [0, .5])
        self.assertEqual(mean.loc[mean.false_positive_rate == 1, "detection_rate"].tolist(), [.75, 1])
