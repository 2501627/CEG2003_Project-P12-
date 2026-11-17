"""Self-contained training of the current learned-alert ANN."""
import copy
import numpy as np
import torch
from torch import nn
KIND = "prelu64_32_16"
WIDTHS = {KIND: (64, 32, 16)}
PARAMETERS = {KIND: 2881}

class DepthANN(nn.Module):
    def __init__(self, kind, init_seed=0):
        super().__init__()
        widths = WIDTHS[kind]
        sizes = (2, *widths, 1)
        modules = []
        for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
            linear = nn.Linear(left, right)
            offset = 101 if index == 0 else 301 if index == len(widths) else 200 + index
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed + offset)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
            modules.append(linear)
            if index < len(widths):
                modules.append(nn.PReLU(right, init=.25) if index == 0 else nn.ReLU())
        self.layers = nn.Sequential(*modules)
        self.output_activation = nn.Sigmoid()
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS[kind]

    def logits(self, x):
        return self.layers(x).squeeze(-1)

    def forward(self, x):
        return self.output_activation(self.logits(x))

def raw_features(preprocessor, rates, log_days):
    result = np.column_stack([np.log1p(rates @ preprocessor["type_weights"]), log_days])
    assert result.shape == (len(rates), 2) and np.isfinite(result).all()
    return result

def validate_training(preprocessor, counts, rates, log_days):
    np.testing.assert_array_equal(preprocessor["training_type_totals"], counts.sum(axis=0))
    x = raw_features(preprocessor, rates, log_days)
    scaler = preprocessor["scaler"]
    assert scaler.n_samples_seen_ == len(x) and scaler.n_features_in_ == 2
    np.testing.assert_allclose(scaler.mean_, x.mean(axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(scaler.var_, x.var(axis=0), rtol=1e-13, atol=1e-13)
    return scaler.transform(x).astype(np.float32)

def state_values(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}

class AlertGate(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        prior = np.asarray(preprocessor["type_weights"], dtype=np.float64)
        assert prior.shape == (10,) and (prior > 0).all()
        self.register_buffer("log_prior", torch.from_numpy(np.log(prior)))
        self.register_buffer("mean", torch.from_numpy(preprocessor["scaler"].mean_.copy()))
        self.register_buffer("scale", torch.from_numpy(preprocessor["scaler"].scale_.copy()))
        self.delta = nn.Parameter(torch.zeros(10, dtype=torch.float64))

    def weights(self):
        return torch.softmax(self.log_prior + self.delta, dim=0) * 10

    def features(self, rates, days):
        raw = torch.stack([torch.log1p(rates @ self.weights()), days], dim=1)
        return ((raw - self.mean) / self.scale).to(torch.float32)

    def penalty(self):
        centered = self.delta - self.delta.mean()
        return (centered * centered).mean()

class GatedANN(nn.Module):
    def __init__(self, preprocessor, seed):
        super().__init__()
        self.network = DepthANN(KIND, seed)
        self.gate = AlertGate(preprocessor)
        assert sum(p.numel() for p in self.parameters()) == 2891

    def logits(self, rates, days):
        return self.network.logits(self.gate.features(rates, days))

    def forward(self, rates, days):
        return self.network.output_activation(self.logits(rates, days))

def snapshot(model):
    return {"state": state_values(model.network), "alert_delta": model.gate.delta.detach().numpy().copy(),
            "learned_type_weights": model.gate.weights().detach().numpy().copy()}

def train_learned(preprocessor, rates, days, labels, seed, epochs, config, regularization, valid=None):
    assert set(np.unique(labels)) == {0, 1}
    torch.manual_seed(seed)
    model = GatedANN(preprocessor, seed)
    initial = state_values(model.network)
    class_weights = len(labels) / (2 * np.bincount(labels, minlength=2))
    tr = torch.from_numpy(np.asarray(rates, dtype=np.float64))
    td = torch.from_numpy(np.asarray(days, dtype=np.float64))
    ty = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    tw = torch.from_numpy(class_weights[np.asarray(labels, dtype=int)].astype(np.float32))
    if valid is not None:
        vr, vd, vy = [torch.from_numpy(np.asarray(value, dtype=np.float64 if i < 2 else np.float32)) for i, value in enumerate(valid)]
        vw = torch.from_numpy(class_weights[np.asarray(valid[2], dtype=int)].astype(np.float32))
    optimizer = torch.optim.RMSprop(model.parameters(), **config["optimizer"])
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    history, states, predictions, weight_history = [], [], [], []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(labels), config["batch_size"]):
            ix = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            bce = (loss_function(model.logits(tr[ix], td[ix]), ty[ix]) * tw[ix]).mean()
            objective = bce + regularization * model.gate.penalty()
            assert bool(torch.isfinite(objective))
            objective.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model.logits(tr, td)
            bce = (loss_function(logits, ty) * tw).mean()
            penalty = regularization * model.gate.penalty()
            item = {"epoch": epoch, "training_weighted_bce": float(bce), "training_penalty": float(penalty),
                    "training_objective": float(bce + penalty),
                    "training_accuracy_at_half": float(((torch.sigmoid(logits) >= .5) == ty.bool()).float().mean())}
            weight_history.append(model.gate.weights().numpy().copy())
            if valid is not None:
                logits = model.logits(vr, vd)
                probabilities = torch.sigmoid(logits)
                item.update({"validation_weighted_bce": float((loss_function(logits, vy) * vw).mean()),
                             "validation_accuracy_at_half": float(((probabilities >= .5) == vy.bool()).float().mean())})
                predictions.append(probabilities.numpy().astype(float))
                states.append(snapshot(model))
        history.append(item)
    result = {"preprocessor": preprocessor, "method": "learned", "kind": KIND, "regularization": regularization,
              "seed": seed, "epochs": epochs, "class_weights": class_weights, "initial_state": initial,
              "initial_type_weights": (torch.softmax(model.gate.log_prior, dim=0) * 10).detach().numpy().copy(),
              "history": history, "weight_history": np.asarray(weight_history), **snapshot(model)}
    if valid is not None:
        result.update({"validation_predictions": np.asarray(predictions), "epoch_states": states})
    assert np.isfinite(result["learned_type_weights"]).all() and (result["learned_type_weights"] > 0).all()
    np.testing.assert_allclose(result["learned_type_weights"].mean(), 1, atol=1e-14)
    return result

def predict(bundle, rates, days):
    model = GatedANN(bundle["preprocessor"], 0)
    model.network.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in bundle["state"].items()}, strict=True)
    with torch.no_grad():
        model.gate.delta.copy_(torch.from_numpy(bundle["alert_delta"]))
    model.eval()
    with torch.inference_mode():
        actual = model.gate.weights().numpy()
        np.testing.assert_array_equal(actual, bundle["learned_type_weights"])
        scores = model(torch.from_numpy(np.asarray(rates, dtype=np.float64)), torch.from_numpy(np.asarray(days, dtype=np.float64))).numpy().astype(float)
    assert scores.shape == (len(rates),) and np.isfinite(scores).all()
    return scores

def select(reference, counts, rates, days, labels, seed, config):
    trials = []
    for regularization in config["regularizations"]:
        models, seen = [], np.zeros(len(labels), dtype=int)
        for inner, partition in enumerate(reference["inner_partitions"]):
            train, valid = np.asarray(partition["train"]), np.asarray(partition["valid"])
            assert set(train).isdisjoint(valid) and set(train) | set(valid) == set(range(len(labels)))
            seen[valid] += 1
            models.append(fit(copy.deepcopy(reference["inner_preprocessors"][inner]), counts[train], rates[train], days[train], labels[train],
                              seed + inner + 1, config["max_epochs"], config, regularization,
                              valid=(rates[valid], days[valid], labels[valid])))
        assert (seen == 1).all()
        losses = np.mean([[h["validation_weighted_bce"] for h in bundle["history"]] for bundle in models], axis=0)
        epoch = int(np.argmin(losses)) + 1
        for bundle in models:
            bundle.update(bundle.pop("epoch_states")[epoch - 1])
            bundle["selected_state_epoch"] = epoch
        trials.append({"regularization": regularization, "inner_models": models, "selected_epochs": epoch,
                       "mean_validation_weighted_bce": losses, "minimum_validation_bce": float(losses[epoch - 1])})
    chosen = min(range(len(trials)), key=lambda i: (trials[i]["minimum_validation_bce"], i))
    best = trials[chosen]
    scores = np.full(len(labels), np.nan)
    for partition, bundle in zip(reference["inner_partitions"], best["inner_models"]):
        scores[np.asarray(partition["valid"])] = bundle["validation_predictions"][best["selected_epochs"] - 1]
    return {"trials": trials, "selected_trial": chosen, "regularization": best["regularization"],
            "inner_models": best["inner_models"], "selected_epochs": best["selected_epochs"], "inner_scores": scores,
            "inner_partitions": copy.deepcopy(reference["inner_partitions"]), "seed": seed, "method": "learned"}

def fit(preprocessor, counts, rates, days, labels, seed, epochs, config, regularization, valid=None):
    validate_training(preprocessor, counts, rates, days)
    return train_learned(preprocessor, rates, days, labels, seed, epochs, config, regularization, valid)

