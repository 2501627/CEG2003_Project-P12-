"""March alert profiles, next-week report labels, and the captain review network."""

import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
ALERT_FILE = DATA / "march_2024_alerts.parquet"
REPORT_FILE = DATA / "march_2024_incidents.parquet"
WEEK = timedelta(days=7)
TRAIN_STARTS = [date(2024, 3, 1), date(2024, 3, 8)]
FINAL_START = date(2024, 3, 18)
TYPES = ["Harsh Braking", "Harsh Acceleration", "Sharp Cornering", "Speeding", "Swaying",
         "Forward Collision Warning", "Pedestrian Collision Alert", "Headway Monitoring Alert",
         "Fatigue Alert", "Yawn"]
COUNTS = [t.lower().replace(" ", "_") for t in TYPES]
RATES = [c + "_per_observed_day" for c in COUNTS]
DAILY_LOG = [c + "_mean_discounted_daily_count" for c in COUNTS]
GROUPS = {
    "braking": ["Harsh Braking"],
    "other_handling": ["Harsh Acceleration", "Sharp Cornering", "Speeding", "Swaying"],
    "proximity": ["Forward Collision Warning", "Headway Monitoring Alert", "Pedestrian Collision Alert"],
    "fatigue_yawn": ["Fatigue Alert", "Yawn"],
}
GROUP_SHARES = [g + "_group_day_share" for g in GROUPS]
FEATURES = [g + "_group_rate" for g in GROUPS] + GROUP_SHARES + ["observed_days"]
TARGET = "reported_next_week"
SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.001
CUTOFF = 0.5


class CaptainANN(nn.Module):
    def __init__(self, input_dim=len(FEATURES)):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(input_dim, 64), nn.PReLU(num_parameters=64),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.output = nn.Sigmoid()

    def forward(self, inputs):
        return self.output(self.logits(inputs)).squeeze(1)


def weekly_profiles(alerts, reports, start):
    """Use one week's alert records for inputs and the following week's reports for labels."""
    days = pl.col("alert_start").dt.date()
    daily = (alerts.filter(days.is_between(start, start + WEEK, closed="left"))
             .group_by("captain_id", days.alias("day"))
             .agg(pl.len().alias("rows"), *[(pl.col("alert_type") == t).sum().alias(c)
                                              for t, c in zip(TYPES, COUNTS)]))
    daily_groups = [
        pl.sum_horizontal([pl.col(COUNTS[TYPES.index(t)]) for t in types])
        for types in GROUPS.values()
    ]
    profiles = daily.group_by("captain_id").agg(
        pl.len().alias("observed_days"), pl.col("rows").sum().alias("alert_rows"),
        *[pl.col(c).mean().alias(rate) for c, rate in zip(COUNTS, RATES)],
        *[(pl.col(c).cast(pl.Float64).log1p() / np.log(2)).mean().alias(out)
          for c, out in zip(COUNTS, DAILY_LOG)],
        *[(group > 0).mean().alias(share) for group, share in zip(daily_groups, GROUP_SHARES)])
    later = reports.filter(pl.col("accident_date").is_between(start + WEEK, start + 2*WEEK, closed="left"))
    labels = later["captain_id"].unique()
    result = (profiles.with_columns(pl.col("captain_id").is_in(labels.implode()).cast(pl.Int8).alias(TARGET),
                                    pl.lit(start).alias("alert_week_start"))
              .sort("captain_id").collect().to_pandas())
    if (result.captain_id.isna().any() or not result.captain_id.is_unique or
            not np.isfinite(result[RATES + DAILY_LOG + GROUP_SHARES]).all().all()):
        raise ValueError("Invalid weekly captain profiles.")
    np.testing.assert_allclose(result[RATES].sum(axis=1) * result.observed_days, result.alert_rows)
    return result


def train_scaled(inputs, y, epochs=EPOCHS):
    """Fit the weekly network to inputs already prepared using training records."""
    x = torch.tensor(inputs, dtype=torch.float32)
    if len(np.unique(y)) != 2:
        raise ValueError("Both label classes are required to train the network.")
    weights = torch.tensor(compute_sample_weight("balanced", y), dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(SEED)
    network = CaptainANN(inputs.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(weight=weights)
    layers = [layer for layer in network.logits if isinstance(layer, nn.Linear)]
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(network.logits(x).squeeze(1), targets)
        loss = loss + sum(layer.weight.square().sum() for layer in layers) / (2 * len(y))
        loss.backward()
        optimizer.step()
    network.eval()
    return network


def ranked(frame):
    ordered = frame.sort_values(["review_score", "captain_id"], ascending=[False, True], ignore_index=True).copy()
    ordered.insert(0, "rank", np.arange(1, len(ordered) + 1))
    ordered["top_ten_percent"] = ordered["rank"] <= (len(ordered) + 9) // 10
    return ordered


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    from training.weighted_weekly_ann import main
    main()
