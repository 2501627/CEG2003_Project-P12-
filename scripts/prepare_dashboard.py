"""Refresh current ANN display tables without checkpoints, legacy models or retraining."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.model_runtime import alert_weights, predict, score_tolerance, validate_bundle
from training.data import digest, load_data

DEST = ROOT / "dashboard/data"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def uncertainty(heldout, ids, labels):
    scores = np.array([heldout.loc[heldout["repeat"] == r].set_index("captain_id").loc[ids, "score"].to_numpy() for r in (1, 2, 3)])
    flags = np.array([heldout.loc[heldout["repeat"] == r].set_index("captain_id").loc[ids, "predicted_report"].to_numpy() for r in (1, 2, 3)])
    positive, negative = np.flatnonzero(labels), np.flatnonzero(labels == 0)
    rng, draws = np.random.default_rng(20261130), []
    for _ in range(2000):
        ix = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        draws.append([np.mean([roc_auc_score(labels[ix], row[ix]) for row in scores]), np.mean(flags[:, ix] == labels[ix][None, :]), flags[:, ix[:len(positive)]].mean()])
    bounds = np.quantile(draws, [.025, .975], axis=0)
    return {key: {"low": float(bounds[0, i]), "high": float(bounds[1, i])} for i, key in enumerate(("roc_auc", "accuracy", "incident_detection_recall"))}


def main():
    data, _, dataset = load_data(verify_sources=True)
    ids, labels, counts, rates, log_days, types = [data[key] for key in ("ids", "labels", "counts", "rates", "log_days", "alert_types")]
    ids, types = ids.tolist(), types.tolist()
    registry_path = ROOT / "dashboard/model_registry.json"
    registry = json.loads(registry_path.read_text())
    registered = registry["learned_alert_ann"]
    registered["artifact_sha256"] = {name: digest(ROOT / name) for name in registered["artifact_sha256"]}
    registered["model_sha256"] = digest(ROOT / registered["model_path"])
    registered["metrics_sha256"] = digest(ROOT / registered["metrics_path"])
    save(registry_path, {"learned_alert_ann": registered})
    model = joblib.load(ROOT / registered["model_path"])
    validate_bundle(model, registered)
    previous_manifest = json.loads((DEST / "manifest.json").read_text())
    for name in ("daily_alerts.csv", "incident_reports.csv"):
        assert digest(DEST / name) == previous_manifest["generated_sha256"][name], name
    daily = pd.read_csv(DEST / "daily_alerts.csv", dtype={"captain_id": str}, float_precision="round_trip")
    daily["day"] = pd.to_datetime(daily["day"])
    totals = daily.pivot_table(index="captain_id", columns="alert_type", values="recorded_alerts", aggfunc="sum", fill_value=0)
    np.testing.assert_array_equal(totals.reindex(index=ids, columns=types).to_numpy(), counts)
    day_totals = daily.groupby(["captain_id", "day"], as_index=False)["recorded_alerts"].sum()
    days = np.rint(np.expm1(log_days)).astype(int)
    np.testing.assert_array_equal(day_totals.groupby("captain_id").size().reindex(ids), days)
    historical = pd.read_csv(ROOT / registered["historical_scores_path"], dtype={"captain_id": str}, float_precision="round_trip").set_index("captain_id")
    scores = historical.loc[ids, "score"].to_numpy()
    np.testing.assert_allclose(predict(model, rates, log_days), scores, **score_tolerance(model))
    weights = alert_weights(model)
    weighted = rates * weights
    recent = day_totals.loc[day_totals["day"] >= "2024-03-25"].groupby("captain_id")["recorded_alerts"].mean()
    earlier = day_totals.loc[day_totals["day"] < "2024-03-25"].groupby("captain_id")["recorded_alerts"].mean()
    captains = pd.DataFrame({"captain_id": ids, "model_score": scores, "above_model_cutoff": scores >= model["classification_cutoff"],
                             "recorded_alerts": counts.sum(axis=1), "observed_days": days, "alerts_per_observed_day": rates.sum(axis=1),
                             "largest_weighted_type": [types[j] for j in weighted.argmax(axis=1)], "april_reported_incident": labels})
    captains["priority_percentile"] = (captains["model_score"].rank(method="average") - 1) / (len(ids) - 1) * 100
    captains["recent_rate_change_pct"] = [100 * (recent[c] / earlier[c] - 1) if c in recent.index and c in earlier.index else np.nan for c in ids]
    captains = captains.sort_values(["model_score", "captain_id"], ascending=[False, True]).reset_index(drop=True)
    captains.insert(1, "rank", np.arange(1, len(ids) + 1))
    breakdown = pd.DataFrame([{"captain_id": captain, "alert_type": name, "recorded_alerts": int(counts[i, j]), "alerts_per_observed_day": float(rates[i, j]),
                               "model_weight": float(weights[j]), "weighted_input": float(weighted[i, j]), "weighted_input_share": float(weighted[i, j] / weighted[i].sum())}
                              for i, captain in enumerate(ids) for j, name in enumerate(types)])
    heldout = pd.read_csv(ROOT / registered["heldout_path"], dtype={"captain_id": str}, float_precision="round_trip")
    assert len(heldout) == 4623 and set(heldout["method"]) == {registered["metrics_method"]}
    metrics = json.loads((ROOT / registered["metrics_path"]).read_text())
    validation = metrics["mean_heldout_ratings"][registered["metrics_method"]]
    old = json.loads((DEST / "summary.json").read_text())
    fields = ("model_captains", "eligible_april_cases", "cleaned_captains", "cleaned_alert_rows", "model_cohort_alert_rows", "model_observed_captain_days",
              "cleaned_incident_reports_jan_may", "april_source_report_rows", "ambiguous_april_report_captains", "predictor_start", "predictor_end", "outcome_start", "outcome_end",
              "missing_id_alert_rows_excluded", "unmatched_incident_rows_excluded", "comparison_targets")
    summary = {key: old[key] for key in fields}
    summary.update({"model_id": "learned_alert_ann", "model_label": registered["label"], "architecture": "64 PReLU → 32 ReLU → 16 ReLU → 1 sigmoid",
                    "hidden_layers": 3, "sigmoid_outputs": 1, "learned_alert_weights": True, "smote": False, "cutoff_policy": "training-selected",
                    "model_cutoff": float(model["classification_cutoff"]), "full_fit_model_flags": int((scores >= model["classification_cutoff"]).sum()),
                    "validation": validation, "validation_repeats": 3, "validation_folds": 5, "uncertainty": uncertainty(heldout, ids, labels),
                    "uncertainty_definition": "Conditional stratified captain bootstrap preserving three predictions per captain; does not account for repeated model exploration.",
                    "targets_met": all(validation[key] >= old["comparison_targets"][key] - 1e-12 for key in old["comparison_targets"]),
                    "score_explanation": "The ANN combines learned weighted alert rate and recorded-day coverage; weights stay fixed until retraining.",
                    "weighted_input_definition": "Learned type weight times original rows per observed alert day, before log transform and ANN."})
    selector = {"model_id": "learned_alert_ann", **{key: registered[key] for key in ("label", "model_path", "model_sha256", "metrics_path", "metrics_method", "metrics_sha256")},
                "score_kind": "Uncalibrated class-balanced ANN score", "ranking_scope": "Historical full-data refit scores; held-out ratings displayed separately."}
    for name, frame in (("captains.csv", captains), ("type_counts.csv", breakdown), ("heldout_predictions.csv", heldout)):
        frame.to_csv(DEST / name, index=False, encoding="utf-8-sig")
    save(DEST / "summary.json", summary)
    save(ROOT / "ACTIVE_MODEL.json", selector)
    upstream = {**registered["artifact_sha256"], **dataset["original_data_sha256"], **dataset["model_inputs_sha256"],
                "data/model_inputs/manifest.json": digest(ROOT / "data/model_inputs/manifest.json"), "dashboard/model_registry.json": digest(registry_path),
                "dashboard/model_runtime.py": digest(ROOT / "dashboard/model_runtime.py")}
    names = ("captains.csv", "type_counts.csv", "heldout_predictions.csv", "summary.json", "daily_alerts.csv", "incident_reports.csv")
    save(DEST / "manifest.json", {"model_id": "learned_alert_ann", "model_label": registered["label"], "selector_sha256": digest(ROOT / "ACTIVE_MODEL.json"),
                                 "generated_sha256": {name: digest(DEST / name) for name in names}, "upstream_sha256": upstream})
    print(json.dumps({"active_model": registered["label"], "captains": len(captains), "validation": validation, "retrained": False}), flush=True)


if __name__ == "__main__":
    main()
