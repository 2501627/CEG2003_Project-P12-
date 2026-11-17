"""End-to-end checks of the standalone learned-alert ANN project."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dashboard.data as source
from dashboard.model_runtime import score_tolerance


def app():
    return AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=45).run()


@unittest.skipUnless(json.loads((ROOT / "ACTIVE_MODEL.json").read_text())["model_id"] == "learned_alert_ann",
                     "These checks apply when the learned-alert ANN is active.")
class DashboardLearnedANNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = source.load_dashboard()

    def test_promoted_parameters_and_cutoff_match_verified_experiment(self):
        import copy
        from training.data import load_data
        from training.model import fit
        dataset, plan, _ = load_data()
        active = self.data["model"]
        settings = json.loads((ROOT / "config/learned_alert_ann.json").read_text())["training"]
        original = fit(copy.deepcopy(plan["full"]["outer_preprocessor"]), dataset["counts"], dataset["rates"], dataset["log_days"], dataset["labels"],
                       active["seed"], active["epochs"], settings, active["regularization"])
        for name in active["state"]:
            np.testing.assert_array_equal(active["state"][name], original["state"][name])
        for name in ("alert_delta", "learned_type_weights"):
            np.testing.assert_array_equal(active[name], original[name])
        self.assertEqual(active["classification_cutoff"], json.loads((ROOT / "config/learned_alert_ann.json").read_text())["cutoff"])
        self.assertEqual(self.data["selector"]["model_id"], "learned_alert_ann")
        self.assertEqual(self.data["selector"]["metrics_method"], "learned_training_cutoff")
        self.assertEqual(self.data["summary"]["hidden_layers"], 3)
        self.assertEqual(self.data["summary"]["sigmoid_outputs"], 1)
        self.assertFalse(self.data["summary"]["smote"])

    def test_all_queue_scores_flags_and_learned_weights_match_saved_artifacts(self):
        data, captains = self.data, self.data["captains"]
        historical = pd.read_csv(ROOT / "outputs/learned_alert_ann/full_fit_historical_scores.csv", dtype={"captain_id": str}, float_precision="round_trip").set_index("captain_id")
        np.testing.assert_array_equal(captains["model_score"], historical.loc[captains["captain_id"], "score"])
        np.testing.assert_array_equal(captains["above_model_cutoff"], captains["model_score"] >= data["summary"]["model_cutoff"])
        np.testing.assert_allclose(source.score_all(data["model"], captains, data["types"]), captains["model_score"], **score_tolerance(data["model"]))
        self.assertEqual(len(captains), 1541)
        self.assertEqual(int(captains["april_reported_incident"].sum()), 36)
        self.assertEqual(int(data["types"]["recorded_alerts"].sum()), 3952554)
        self.assertTrue(captains["model_score"].is_monotonic_decreasing)
        weights = data["types"].pivot(index="captain_id", columns="alert_type", values="model_weight").reindex(columns=data["model"]["alert_types"])
        np.testing.assert_allclose(weights, np.broadcast_to(data["model"]["learned_type_weights"], weights.shape), atol=1e-14)
        self.assertGreater(np.max(np.abs(data["model"]["learned_type_weights"] - data["model"]["preprocessor"]["type_weights"])), 1e-5)

    def test_validation_uses_only_the_correct_saved_heldout_policy(self):
        values = []
        self.assertEqual(set(self.data["heldout"]["method"]), {"learned_training_cutoff"})
        for _, group in self.data["heldout"].groupby("repeat"):
            self.assertEqual(len(group), 1541)
            self.assertEqual(group["captain_id"].nunique(), 1541)
            self.assertEqual(int(group["actual_april_report"].sum()), 36)
            values.append([roc_auc_score(group["actual_april_report"], group["score"]), accuracy_score(group["actual_april_report"], group["predicted_report"]), recall_score(group["actual_april_report"], group["predicted_report"])])
        expected = self.data["summary"]["validation"]
        np.testing.assert_allclose(np.mean(values, axis=0), [expected["roc_auc"], expected["accuracy"], expected["incident_detection_recall"]], atol=1e-14)
        self.assertFalse(self.data["summary"]["targets_met"])

    def test_nonlinear_explanations_add_up_to_score_change(self):
        for index in (0, 700, 1540):
            row = self.data["captains"].iloc[index]
            result = source.captain_inference(self.data, row["captain_id"])
            self.assertEqual(result["explanation_kind"], "reference_score_change")
            self.assertAlmostEqual(result["baseline_score"] + result["weighted_alert_input_effect"] + result["observed_days_effect"], result["score"], places=12)
            np.testing.assert_allclose(result["score"], row["model_score"], **score_tolerance(self.data["model"]))

    def test_default_app_shows_one_active_ann_and_its_actual_ratings(self):
        at = app()
        self.assertFalse(at.exception)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["ROC-AUC"], "0.601")
        self.assertEqual(metrics["Binary accuracy"], "66.06%")
        self.assertEqual(metrics["Incident detection"], "52.78%")
        self.assertEqual(len(at.dataframe[0].value), 155)
        self.assertTrue(any(x.value == "Learned-alert ANN model validation" for x in at.subheader))
        self.assertTrue(any("3 hidden layers" in x.value and "64 PReLU" in x.value for x in at.markdown))
        self.assertTrue(any("Original cleaned data preserved" in x.value for x in at.caption))
        self.assertTrue(any("Does not meet all" in x.value and "66.47%" in x.value for x in at.caption))

    def test_search_select_and_verify_actual_ann_score(self):
        at = app()
        captain = self.data["captains"].iloc[-1]["captain_id"]
        at.selectbox(key="queue_scope").select("All captains").run()
        at.text_input(key="captain_search").set_value(captain).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="detail_captain").value, captain)
        at.button(key="verify_score").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("matches the displayed score" in x.value for x in at.success))

    def test_capacity_and_cutoff_filters_do_not_change_ratings(self):
        at = app()
        at.selectbox(key="queue_scope").select("Above model cutoff").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.dataframe[0].value), int(self.data["captains"]["above_model_cutoff"].sum()))
        at.selectbox(key="queue_scope").select("Review capacity").run()
        at.slider(key="review_capacity").set_value(5).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.dataframe[0].value), 78)
        self.assertEqual({m.label: m.value for m in at.metric}["Binary accuracy"], "66.06%")

    def test_missing_search_is_handled(self):
        at = app()
        at.text_input(key="captain_search").set_value("NO-SUCH-CAPTAIN").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("No captains match" in x.value for x in at.info))

    def test_changed_model_is_rejected_without_writing_files(self):
        original = source.digest
        model_path = ROOT / self.data["selector"]["model_path"]
        def changed(path):
            return "0" * 64 if Path(path) == model_path else original(path)
        st.cache_data.clear()
        try:
            with patch.object(source, "digest", changed):
                at = app()
                self.assertFalse(at.exception)
                self.assertTrue(any("Source or model changed" in x.value for x in at.error))
        finally:
            st.cache_data.clear()

    def test_original_data_and_standalone_project_integrity(self):
        from training.data import load_data
        load_data(verify_sources=True)
        for name in ("checkpoints", "experiments", "rollback_backups"):
            self.assertFalse((ROOT / name).exists())
        for name in ("gate_model", "depth_model", "ann_model"):
            self.assertNotIn(name, sys.modules, "Production inference must not import experiment training code.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
