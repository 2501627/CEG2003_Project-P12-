"""Monthly alert totals shared by current and already-running dashboard sessions."""

from datetime import date

import polars as pl


def summarize_month(alerts, start, weights):
    month = pl.col("alert_start").dt.date().dt.month_start()
    counts = (alerts.filter(month == start.replace(day=1))
              .group_by("alert_type").agg(pl.len().alias("month_records")).collect().to_pandas())
    # Keep an alert type visible even when it has a training weight and zero records this month.
    summary = counts.merge(weights[["alert_type", "weight"]], on="alert_type",
                           how="outer", validate="one_to_one")
    summary["month_records"] = summary.month_records.fillna(0).astype("int64")
    return summary.sort_values(["month_records", "alert_type"], ascending=[False, True], ignore_index=True)


def display_alert_summary(snapshot, alert_file):
    summary = getattr(snapshot, "alert_summary", None)
    if summary is None:
        # A running Streamlit server can retain the older loader that returns only training weights.
        # That loader also verifies the cleaned alert file and the weights against the final run.
        start = date.fromisoformat(snapshot.report["alert_week_start"])
        summary = summarize_month(pl.scan_parquet(alert_file), start, snapshot.weights)
    return summary
