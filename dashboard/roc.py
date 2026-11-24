"""Empirical ROC coordinates for one fold or the equally weighted fold mean."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def roc_points(predictions, fold=None):
    rows = predictions if fold is None else predictions.loc[predictions.evaluation_fold == fold]
    curves = [roc_curve(part.reported_next_week, part.review_score)[:2]
              for _, part in rows.groupby("evaluation_fold")]
    if fold is not None:
        x, y = curves[0]
    else:
        # Average at every fold's breakpoints, retaining both ends of vertical jumps.
        grid = np.unique(np.concatenate([fpr for fpr, _ in curves]))
        heights = []
        for fpr, tpr in curves:
            # At a repeated false-positive rate, interpolation gives the top of the jump.
            # The first detection rate at that point supplies the bottom for the plotted line.
            upper = np.interp(grid, fpr, tpr)
            lower = upper.copy()
            knots, first = np.unique(fpr, return_index=True)
            lower[np.searchsorted(grid, knots)] = tpr[first]
            heights.append(np.column_stack((lower, upper)))
        x = np.repeat(grid, 2)
        y = np.mean(heights, axis=0).ravel()
    points = pd.DataFrame({"false_positive_rate": x, "detection_rate": y}).drop_duplicates(ignore_index=True)
    return points.assign(point=points.index)
