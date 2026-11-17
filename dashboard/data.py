"""Guard dashboard provenance and execute the explicitly selected registered classifier."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dashboard.model_config import active_selection
from dashboard.model_runtime import alert_weights, explain, predict, score_tolerance, validate_bundle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard/data"


class DashboardDataError(ValueError):
    pass


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def fingerprint():
    manifest_path = DATA / "manifest.json"
    if not manifest_path.exists():
        raise DashboardDataError("Dashboard data is not prepared. Run python -B scripts/prepare_dashboard.py from the project root.")
    manifest = json.loads(manifest_path.read_text())
    paths = [manifest_path, ROOT / "ACTIVE_MODEL.json", ROOT / "dashboard/model_registry.json", ROOT / "dashboard/model_runtime.py"]
    paths.extend(local_path(name) for name in manifest["upstream_sha256"])
    paths.extend(DATA / name for name in manifest["generated_sha256"])
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(set(paths)))


def load_dashboard():
    manifest = json.loads((DATA / "manifest.json").read_text())
    selector, registered = active_selection()
    model_path = selector["model_path"]
    if selector["model_id"] != manifest["model_id"] or selector["label"] != manifest["model_label"]:
        raise DashboardDataError("The active model selection does not match the prepared dashboard.")
    if digest(ROOT / "ACTIVE_MODEL.json") != manifest["selector_sha256"]:
        raise DashboardDataError("The active model selection does not match the prepared dashboard.")
    for name, expected in manifest["upstream_sha256"].items():
        if digest(local_path(name)) != expected:
            raise DashboardDataError(f"Source or model changed: {name}. Rebuild the dashboard from the verified saved model files.")
    for name, expected in manifest["generated_sha256"].items():
        if Path(name).name != name or digest(DATA / name) != expected:
            raise DashboardDataError("Prepared dashboard files changed. Rebuild the dashboard data.")
    if selector["model_sha256"] != manifest["upstream_sha256"][model_path] or selector["metrics_sha256"] != manifest["upstream_sha256"][selector["metrics_path"]]:
        raise DashboardDataError("The prepared model or ratings do not match the selected model version.")
    bundle = joblib.load(ROOT / model_path)
    validate_bundle(bundle, registered)
    summary = json.loads((DATA / "summary.json").read_text())
    captains = pd.read_csv(DATA / "captains.csv", dtype={"captain_id": str}, float_precision="round_trip")
    daily = pd.read_csv(DATA / "daily_alerts.csv", dtype={"captain_id": str}, float_precision="round_trip")
    daily["day"] = pd.to_datetime(daily["day"])
    types = pd.read_csv(DATA / "type_counts.csv", dtype={"captain_id": str}, float_precision="round_trip")
    incidents = pd.read_csv(DATA / "incident_reports.csv", dtype=str, keep_default_na=False)
    heldout = pd.read_csv(DATA / "heldout_predictions.csv", dtype={"captain_id": str}, float_precision="round_trip")
    if len(captains) != 1541 or captains["captain_id"].nunique() != 1541 or int(captains["april_reported_incident"].sum()) != 36:
        raise DashboardDataError("Dashboard cohort does not match the saved evaluation.")
    if summary["model_id"] != selector["model_id"] or summary["model_label"] != selector["label"]:
        raise DashboardDataError("Displayed model label does not match the selected version.")
    metrics = json.loads((ROOT / selector["metrics_path"]).read_text())["mean_heldout_ratings"][selector["metrics_method"]]
    if summary["validation"] != metrics:
        raise DashboardDataError("Displayed validation metrics do not match the selected saved model.")
    expected_weights = types.pivot(index="captain_id", columns="alert_type", values="model_weight").reindex(columns=bundle["alert_types"])
    np.testing.assert_allclose(expected_weights, np.broadcast_to(alert_weights(bundle), expected_weights.shape), rtol=1e-13, atol=1e-14)
    if len(heldout) != 4623 or set(heldout["method"]) != {selector["metrics_method"]}:
        raise DashboardDataError("The held-out curve does not match the active model's validation method.")
    rescored = score_all(bundle, captains, types)
    np.testing.assert_allclose(rescored, captains["model_score"], **score_tolerance(bundle))
    np.testing.assert_array_equal(captains["above_model_cutoff"], captains["model_score"] >= bundle["classification_cutoff"])
    return {"summary": summary, "captains": captains, "daily": daily, "types": types,
            "incidents": incidents, "heldout": heldout, "model": bundle, "selector": selector}


def score_all(bundle, captains, types):
    pivot = types.pivot(index="captain_id", columns="alert_type", values="recorded_alerts").reindex(index=captains["captain_id"], columns=bundle["alert_types"])
    rates = pivot.to_numpy(dtype=float) / captains["observed_days"].to_numpy()[:, None]
    return predict(bundle, rates, np.log1p(captains["observed_days"].to_numpy()))


def captain_inference(data, captain):
    row = data["captains"].loc[data["captains"]["captain_id"] == captain].iloc[0]
    ordered = data["types"].loc[data["types"]["captain_id"] == captain].set_index("alert_type").reindex(data["model"]["alert_types"])
    return explain(data["model"], ordered["alerts_per_observed_day"].to_numpy(dtype=float), np.log1p(row["observed_days"]))


def filter_queue(captains, scope, capacity, search):
    ranked = captains.sort_values(["rank", "captain_id"])
    if scope == "Review capacity":
        ranked = ranked.head(max(1, int(np.ceil(len(captains) * capacity / 100))))
    elif scope == "Above model cutoff":
        ranked = ranked.loc[ranked["above_model_cutoff"]]
    if search.strip():
        ranked = ranked.loc[ranked["captain_id"].str.contains(search.strip(), regex=False)]
    return ranked.copy()
