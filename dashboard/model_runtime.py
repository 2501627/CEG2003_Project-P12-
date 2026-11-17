"""Inference-only runtime for the active learned-alert ANN."""
import numpy as np


def is_learned_ann(bundle):
    return bundle.get("method") == "learned" and bundle.get("kind") == "prelu64_32_16" and "alert_delta" in bundle


def alert_weights(bundle):
    assert is_learned_ann(bundle)
    return bundle["learned_type_weights"]


def score_tolerance(bundle):
    assert is_learned_ann(bundle)
    return {"rtol": 5e-6, "atol": 2e-6}


def ann_runtime(bundle):
    import torch
    from torch import nn
    torch.set_num_threads(1)

    class RuntimeANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(2, 64), nn.PReLU(64), nn.Linear(64, 32), nn.ReLU(),
                                        nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            self.output_activation = nn.Sigmoid()

        def forward(self, features):
            return self.output_activation(self.layers(features).squeeze(-1))

    assert is_learned_ann(bundle)
    model = RuntimeANN()
    model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in bundle["state"].items()}, strict=True)
    assert sum(p.numel() for p in model.parameters()) == 2881
    p = bundle["preprocessor"]
    log_prior = torch.from_numpy(np.log(p["type_weights"]))
    delta = torch.from_numpy(bundle["alert_delta"])
    learned = torch.softmax(log_prior + delta, dim=0) * 10
    np.testing.assert_array_equal(learned.numpy(), bundle["learned_type_weights"])
    return model.eval(), learned


def validate_bundle(bundle, registered):
    if registered.get("family") == "learned_alert_ann":
        assert is_learned_ann(bundle) and bundle["preprocessor"]["scaler"].n_features_in_ == 2
        assert len(bundle["alert_types"]) == 10 and bundle["learned_type_weights"].shape == (10,)
        assert (bundle["learned_type_weights"] > 0).all()
        np.testing.assert_allclose(bundle["learned_type_weights"].mean(), 1, atol=1e-14)
        ann_runtime(bundle)
    else:
        raise ValueError("Only the current learned-alert ANN is supported.")


def predict(bundle, rates, log_days):
    assert is_learned_ann(bundle)
    import torch
    model, weights = ann_runtime(bundle)
    p = bundle["preprocessor"]
    with torch.inference_mode():
        raw = torch.stack([torch.log1p(torch.from_numpy(np.asarray(rates, dtype=np.float64)) @ weights),
                           torch.from_numpy(np.asarray(log_days, dtype=np.float64))], dim=1)
        features = ((raw - torch.from_numpy(p["scaler"].mean_)) / torch.from_numpy(p["scaler"].scale_)).to(torch.float32)
        result = model(features).numpy().astype(float)
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result


def explain(bundle, rates, log_days):
    raw = np.array([[np.log1p(rates @ alert_weights(bundle)), log_days]], dtype=float)
    assert is_learned_ann(bundle)
    import torch
    model, _ = ann_runtime(bundle)
    scaler = bundle["preprocessor"]["scaler"]
    reference = scaler.mean_
    # Exact two-input Shapley effects relative to the frozen scaler's reference inputs.
    combinations = np.array([reference, [raw[0, 0], reference[1]], [reference[0], raw[0, 1]], raw[0]])
    with torch.inference_mode():
        values = model(torch.from_numpy(scaler.transform(combinations).astype(np.float32))).numpy().astype(float)
    alert_effect = .5 * ((values[1] - values[0]) + (values[3] - values[2]))
    day_effect = .5 * ((values[2] - values[0]) + (values[3] - values[1]))
    return {"score": float(values[3]), "baseline_score": float(values[0]), "weighted_alert_input_effect": float(alert_effect),
            "observed_days_effect": float(day_effect), "explanation_kind": "reference_score_change"}
