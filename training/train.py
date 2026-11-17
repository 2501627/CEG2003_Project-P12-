"""Retrain the current ANN; optionally repeat its saved captain-fold evaluation."""
import argparse
import copy
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits

from training.data import ROOT, load_data
from training.model import fit, predict, select


def cutoff(labels, scores, minimum):
    candidates = []
    for value in np.unique(scores):
        flags = scores >= value
        if flags[labels == 1].mean() >= minimum - 1e-12:
            candidates.append((float(np.mean(flags == labels)), float(value)))
    return max(candidates)[1]


def ratings(labels, scores, thresholds):
    flags = scores >= thresholds
    return {"roc_auc": float(roc_auc_score(labels, scores)), "accuracy": float(accuracy_score(labels, flags)),
            "incident_detection_recall": float(recall_score(labels, flags)), "precision": float(precision_score(labels, flags, zero_division=0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Repeat the original 3 x 5 captain tests, with training-only selection.")
    parser.add_argument("--output", default="outputs/retrained_ann", help="A new output directory inside the project; never overwrites the active model.")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    destination.relative_to(ROOT)
    assert not destination.exists(), "Choose a new output directory."
    config = json.loads((ROOT / "config/learned_alert_ann.json").read_text())
    settings = config["training"]
    data, plan, manifest = load_data(verify_sources=True)
    labels, counts, rates, days = [data[key] for key in ("labels", "counts", "rates", "log_days")]
    destination.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.evaluate:
            scores, thresholds = np.full((3, len(labels)), np.nan), np.full((3, len(labels)), np.nan)
            audits = []
            for group in plan["folds"]:
                train, test = np.asarray(group["train_indices"]), np.asarray(group["test_indices"])
                assert set(train).isdisjoint(test) and set(train) | set(test) == set(range(len(labels)))
                seed = settings["seed"] + group["repeat"] * 1000 + group["fold"] * 10
                chosen = select(group, counts[train], rates[train], days[train], labels[train], seed, settings)
                threshold = cutoff(labels[train], chosen["inner_scores"], settings["minimum_training_recall"])
                model = fit(copy.deepcopy(group["outer_preprocessor"]), counts[train], rates[train], days[train], labels[train],
                            seed + 4, chosen["selected_epochs"], settings, chosen["regularization"])
                scores[group["repeat"] - 1, test] = predict(model, rates[test], days[test])
                thresholds[group["repeat"] - 1, test] = threshold
                audits.append({"repeat": group["repeat"], "fold": group["fold"], "epochs": chosen["selected_epochs"], "regularization": chosen["regularization"], "cutoff": threshold})
                print(json.dumps(audits[-1]), flush=True)
            assert np.isfinite(scores).all() and np.isfinite(thresholds).all()
            results = {}
            for policy, values in (("training_selected", thresholds), ("fixed_half", np.full_like(thresholds, .5))):
                repeats = [ratings(labels, scores[i], values[i]) for i in range(3)]
                results[policy] = {key: float(np.mean([row[key] for row in repeats])) for key in repeats[0]}
            (destination / "metrics.json").write_text(json.dumps({"mean_heldout_ratings": results, "fold_selection": audits, "cohort": manifest["cohort"]}, indent=2), encoding="utf-8")
            with (destination / "heldout_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["repeat", "captain_id", "actual_april_report", "score", "training_cutoff"])
                for repeat in range(3):
                    writer.writerows((repeat + 1, captain, int(labels[i]), float(scores[repeat, i]), float(thresholds[repeat, i])) for i, captain in enumerate(data["ids"]))
            full = select(plan["full"], counts, rates, days, labels, settings["seed"] + 20000, settings)
            epochs, regularization = full["selected_epochs"], full["regularization"]
            threshold = cutoff(labels, full["inner_scores"], settings["minimum_training_recall"])
        else:
            epochs, regularization, threshold = config["full_refit_epochs"], config["selected_regularization"], config["cutoff"]
        model = fit(copy.deepcopy(plan["full"]["outer_preprocessor"]), counts, rates, days, labels,
                    settings["seed"] + 20004, epochs, settings, regularization)
        model.update({"alert_types": data["alert_types"].tolist(), "type_weights": model["learned_type_weights"].copy(),
                      "model_family": "learned_alert_ann", "classification_cutoff": threshold, "config": config})
        joblib.dump(model, destination / "classifier.joblib")
    print(json.dumps({"saved_to": destination.relative_to(ROOT).as_posix(), "epochs": epochs, "regularization": regularization,
                      "active_model_changed": False, "evaluated": args.evaluate}), flush=True)


if __name__ == "__main__":
    main()
