"""Load the current model's numeric inputs and original training/test groups."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_data(verify_sources=False):
    manifest = json.loads((ROOT / "data/model_inputs/manifest.json").read_text())
    hashes = dict(manifest["model_inputs_sha256"])
    if verify_sources:
        hashes.update(manifest["original_data_sha256"])
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    with np.load(ROOT / "data/model_inputs/cohort.npz", allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    assert len(data["ids"]) == 1541 and data["labels"].sum() == 36 and data["counts"].sum() == 3952554
    plan = joblib.load(ROOT / "data/model_inputs/training_plan.joblib")
    return data, plan, manifest
