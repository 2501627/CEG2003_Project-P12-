"""Allow only explicitly registered and verified model versions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def registered_model(model_id):
    registry = json.loads((ROOT / "dashboard/model_registry.json").read_text())
    if model_id != "learned_alert_ann" or model_id not in registry:
        raise ValueError("The dashboard model must be an explicitly registered model version.")
    return registry[model_id]


def active_selection():
    selection = json.loads((ROOT / "ACTIVE_MODEL.json").read_text())
    registered = registered_model(selection["model_id"])
    for key in ("label", "model_path", "model_sha256", "metrics_path", "metrics_method", "metrics_sha256"):
        if selection[key] != registered[key]:
            raise ValueError("The active model does not match its registered model version.")
    return selection, registered
