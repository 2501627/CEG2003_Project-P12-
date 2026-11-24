"""Read the active March model's published results and its matching cleaned alerts."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

from dashboard.monthly_alerts import summarize_month


ROOT = Path(__file__).resolve().parents[1]
FINAL = Path("outputs/final")
REPORT = FINAL / "results.json"
QUEUE = FINAL / "march_25_31_review_list.csv"
PREDICTIONS = FINAL / "evaluation_predictions.csv"
WEIGHTS = FINAL / "alert_weights.csv"
ALERTS = Path("data/processed/march_2024_alerts.parquet")
INPUTS = (REPORT, QUEUE, PREDICTIONS, WEIGHTS, ALERTS)
WEEK = timedelta(days=7)


@dataclass
class Snapshot:
    report: dict
    queue: pd.DataFrame
    counts: pd.DataFrame
    predictions: pd.DataFrame
    alert_summary: pd.DataFrame


def fingerprint(root):
    stamp = []
    for relative in INPUTS:
        info = (root / relative).stat()
        stamp.append((info.st_size, info.st_mtime_ns))
    return tuple(stamp)


def load_snapshot(root=ROOT):
    report = json.loads((root / REPORT).read_text(encoding="utf-8"))
    for relative in (QUEUE, PREDICTIONS, WEIGHTS, ALERTS):
        hashes = report["source_sha256"] if relative == ALERTS else report["artifact_sha256"]
        with (root / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[relative.name]:
                raise ValueError(f"{relative.name} differs from the file recorded by the final model.")

    queue = pd.read_csv(root / QUEUE, float_precision="round_trip").sort_values("rank", ignore_index=True)
    predictions = pd.read_csv(root / PREDICTIONS, float_precision="round_trip")
    weights = pd.read_csv(root / WEIGHTS, float_precision="round_trip")
    start = date.fromisoformat(report["alert_week_start"])
    end = date.fromisoformat(report["outcome_week_start"])
    if end - start != WEEK:
        raise ValueError("The saved alert and outcome weeks do not align.")
    day = pl.col("alert_start").dt.date()
    alerts = pl.scan_parquet(root / ALERTS)
    # The totals table covers the whole month. The queue and each captain's detail use the alert week.
    alert_summary = summarize_month(alerts, start, weights)
    daily = (alerts
             .filter(day.is_between(start, end, closed="left"))
             .group_by("captain_id", day.alias("day"), "alert_type")
             .agg(pl.len().alias("records")).collect().to_pandas())
    counts = daily.groupby(["captain_id", "alert_type"]).records.sum().unstack(fill_value=0)
    days = daily.groupby("captain_id").day.nunique()
    ids = queue.captain_id
    # Join by captain ID because the queue is rank-ordered and the count table is grouped by ID.
    if (not ids.is_unique or set(ids) != set(counts.index) or
            not predictions.captain_id.is_unique or set(ids) != set(predictions.captain_id) or
            not days.reindex(ids).eq(queue.set_index("captain_id").observed_days).all() or
            not queue.flagged.eq(queue.review_score >= report["cutoff"]).all() or
            not set(counts.columns).issubset(weights.alert_type)):
        raise ValueError("The alert profiles, evaluation, and review list do not describe the same run.")
    counts = counts.reindex(columns=sorted(weights.alert_type), fill_value=0)
    queue["records"] = ids.map(counts.sum(axis=1))
    return Snapshot(report, queue, counts, predictions, alert_summary)
