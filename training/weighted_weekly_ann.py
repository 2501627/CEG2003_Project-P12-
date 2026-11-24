"""Train and evaluate the grouped March ANN with automatic alert-frequency weights."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from training import weekly_ann as base

N_FOLDS = 5
ARTIFACTS = ("weighted_ann.pt", "alert_weights.csv", "evaluation_predictions.csv", "march_25_31_review_list.csv")


def fit_alert_weights(training):
    """Fit one multiplier per alert from training counts and observed alert days."""
    exposure = training.observed_days.sum()
    if exposure <= 0 or training.duplicated(["captain_id", "alert_week_start"]).any():
        raise ValueError("Expected distinct, observed captain-weeks in the training data.")
    totals = np.rint(training[base.RATES].multiply(training.observed_days, axis=0).sum()).astype("int64")
    per_day = totals.to_numpy() / exposure
    return pd.DataFrame({"alert_type": base.TYPES, "training_rows": totals.to_numpy(),
                         "rows_per_observed_captain_day": per_day,
                         "weight": 1 / np.sqrt(1 + per_day)})


def weighted_inputs(frame, alert_weights):
    weights = np.asarray(alert_weights, dtype=np.float64)
    if weights.shape != (len(base.TYPES),) or not np.all((weights > 0) & (weights <= 1)):
        raise ValueError("Expected one positive multiplier at most one for each known alert type.")
    discounted = frame[base.DAILY_LOG].to_numpy(dtype=np.float64) * weights
    group_values = [discounted[:, [base.TYPES.index(t) for t in types]].sum(axis=1)
                    for types in base.GROUPS.values()]
    return np.column_stack([*group_values, frame[base.GROUP_SHARES], frame.observed_days])


def fit_weighted(training):
    weights = fit_alert_weights(training)
    values = weighted_inputs(training, weights.weight)
    center = values.mean(axis=0)
    # Scaling by the unweighted training spread preserves attenuation even in
    # a one-type group (harsh braking). Scaling by its weighted spread would undo it.
    reference = weighted_inputs(training, np.ones(len(base.TYPES)))
    scale = StandardScaler().fit(reference).scale_
    net = base.train_scaled((values - center) / scale, training[base.TARGET].to_numpy(dtype=int))
    bundle = {"model_state": net.state_dict(), "alert_types": base.TYPES, "groups": base.GROUPS,
              "features": base.FEATURES, "weights": torch.tensor(weights.weight.to_numpy()),
              "weighted_center": torch.tensor(center), "reference_scale": torch.tensor(scale),
              "daily_transform": "log2(1+count)", "cutoff": base.CUTOFF}
    return bundle, weights


@torch.inference_mode()
def score_weighted(frame, bundle):
    if bundle["alert_types"] != base.TYPES or bundle["groups"] != base.GROUPS or bundle["features"] != base.FEATURES:
        raise ValueError("Saved alert feature schema differs from scoring schema.")
    values = weighted_inputs(frame, bundle["weights"].numpy())
    scaled = (values - bundle["weighted_center"].numpy()) / bundle["reference_scale"].numpy()
    network = base.CaptainANN(scaled.shape[1])
    network.load_state_dict(bundle["model_state"])
    scores = network.eval()(torch.tensor(scaled, dtype=torch.float32)).numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite weighted ANN scores.")
    return scores


def final_ratings(predictions):
    """Give each captain fold equal weight, using the fixed cutoff for accuracy."""
    ids = predictions["captain_id"]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("Expected one evaluation prediction per captain.")
    folds = predictions["evaluation_fold"]
    if set(folds.unique()) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Expected five nonempty evaluation folds.")
    if not predictions["review_score"].between(0, 1).fillna(False).all():
        raise ValueError("Every captain must have a finite score between zero and one.")
    aucs, accuracies = [], []
    for _, part in predictions.groupby("evaluation_fold", sort=True):
        y, scores = part[base.TARGET], part["review_score"]
        if set(y.unique()) != {0, 1}:
            raise ValueError("Each fold needs captains with and without a report to calculate ROC-AUC.")
        aucs.append(roc_auc_score(y, scores))
        accuracies.append(accuracy_score(y, scores >= base.CUTOFF))
    return {"roc_auc": float(np.mean(aucs)), "accuracy": float(np.mean(accuracies))}


def evaluate(history, final):
    people = pd.Series(sorted(set(history.captain_id) | set(final.captain_id)))
    # Use final labels only to balance the evaluation groups. -1 means no final-week profile.
    strata = people.map(final.set_index("captain_id")[base.TARGET]).fillna(-1).astype(int)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=base.SEED)
    ledgers = []
    for fold, (fit_index, check_index) in enumerate(splitter.split(people, strata), 1):
        train_rows = history.loc[history.captain_id.isin(people.iloc[fit_index])]
        check_rows = final.loc[final.captain_id.isin(people.iloc[check_index])]
        if set(train_rows.captain_id).intersection(check_rows.captain_id):
            raise ValueError("A scored captain also appears in that fold's training data.")
        bundle, _ = fit_weighted(train_rows)
        ledger = check_rows[["captain_id", base.TARGET]].copy()
        ledger["review_score"] = score_weighted(check_rows, bundle)
        ledger["evaluation_fold"] = fold
        ledgers.append(ledger)
    predictions = pd.concat(ledgers, ignore_index=True).sort_values("captain_id")
    if set(predictions.captain_id) != set(final.captain_id):
        raise ValueError("Evaluation does not cover all final-week captains.")
    return predictions


def run(dest):
    sources = (base.ALERT_FILE, base.REPORT_FILE, Path(__file__), Path(base.__file__))
    hashes = {p.name: base.file_hash(p) for p in sources}
    if dest.exists():
        report = json.loads((dest / "results.json").read_text(encoding="utf8"))
        if report.get("source_sha256") != hashes:
            raise ValueError("Inputs or training code have changed. Use --output with a new directory to refit.")
        if report.get("artifact_sha256") != {name: base.file_hash(dest / name) for name in ARTIFACTS}:
            raise ValueError("Saved files differ from the verified run. Refit in a new output directory.")
        predictions = pd.read_csv(dest / "evaluation_predictions.csv", float_precision="round_trip")
        if (len(predictions) != report["evaluated_captains"] or
                int(predictions[base.TARGET].sum()) != report["incident_captains"] or
                final_ratings(predictions) != report["ratings"]):
            raise ValueError("Saved predictions do not reproduce the final evaluation.")
        return report["ratings"]

    alerts = pl.scan_parquet(base.ALERT_FILE)
    raw = pl.read_parquet(base.REPORT_FILE)
    duplicate = pl.col("source_annotation").fill_null("").str.strip_chars().str.to_lowercase() == "duplicate"
    reports = raw.filter(~duplicate & pl.col("has_march_alerts"))
    if reports["report_id"].n_unique() != len(reports) or reports["captain_id"].null_count():
        raise ValueError("Eligible reports must have distinct report IDs and a known captain.")
    if max(base.TRAIN_STARTS) + 2 * base.WEEK > base.FINAL_START + base.WEEK:
        raise ValueError("Training outcomes must be known before the final prediction week.")
    history = pd.concat([base.weekly_profiles(alerts, reports, start) for start in base.TRAIN_STARTS],
                        ignore_index=True)
    final = base.weekly_profiles(alerts, reports, base.FINAL_START)
    predictions = evaluate(history, final)
    ratings = final_ratings(predictions)

    # Produce one deployable model using the earlier weeks only. Its scores are for the queue.
    review_bundle, review_weights = fit_weighted(history)
    queue = final[["captain_id", "observed_days"]].copy()
    queue["review_score"] = score_weighted(final, review_bundle)
    queue["flagged"] = queue.review_score >= base.CUTOFF
    flagged = predictions["review_score"] >= base.CUTOFF
    report = {
        "ratings": ratings, "evaluation": "Unweighted mean of five captain-disjoint fold ROC-AUCs and accuracies.",
        "model": "Automatically frequency-weighted grouped March ANN",
        "architecture": "64 PReLU -> 32 ReLU -> 16 ReLU -> 1 sigmoid",
        "folds": N_FOLDS, "seed": base.SEED, "epochs": base.EPOCHS, "cutoff": base.CUTOFF,
        "evaluated_captains": len(predictions), "incident_captains": int(predictions[base.TARGET].sum()),
        "evaluation_flagged": int(flagged.sum()),
        "evaluation_incident_captains_flagged": int(predictions.loc[flagged, base.TARGET].sum()),
        "training_alert_week_starts": [str(s) for s in base.TRAIN_STARTS],
        "alert_week_start": str(base.FINAL_START), "outcome_week_start": str(base.FINAL_START + base.WEEK),
        "review_model_fit": "All eligible earlier training profiles, with outcomes through March 21.",
        "review_use": "Ranks captains for human review by next-week reported-incident score.",
        "source_sha256": hashes,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        torch.save(review_bundle, stage / "weighted_ann.pt")
        review_weights.sort_values("training_rows", ascending=False).to_csv(stage / "alert_weights.csv", index=False)
        predictions.to_csv(stage / "evaluation_predictions.csv", index=False)
        base.ranked(queue).to_csv(stage / "march_25_31_review_list.csv", index=False)
        report["artifact_sha256"] = {name: base.file_hash(stage / name) for name in ARTIFACTS}
        (stage / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        stage.rename(dest)
    return ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=base.ROOT / "outputs/final")
    dest = parser.parse_args().output.resolve()
    torch.set_num_threads(1)
    ratings = run(dest)
    print(f"Mean five-fold ROC-AUC {ratings['roc_auc']:.4f} | Mean five-fold accuracy {ratings['accuracy']:.2%}")


if __name__ == "__main__":
    main()
