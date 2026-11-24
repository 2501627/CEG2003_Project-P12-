# March captain safety review

The model uses one week of a captain's March 2024 driving alerts to rank captains for human review in the following week. A positive label means the captain had at least one eligible incident report in that following week. A flag alone does not establish unsafe driving or fault.

## Run the project

Git clones include the cleaned Parquet files used by the dashboard and model. To download the original Excel workbooks as well, install Git LFS and run `git lfs pull` from the project root.

From the project root, install the dependencies with `python -m pip install -r training/requirements.txt`, then run `python -m training`. The command verifies the saved results, or trains from the two cleaned Parquet files in `data/processed` if `outputs/final` is absent. To reproduce training in a new folder, use `python -m training --output outputs/reproduction`.

## Open the local dashboard

In the VS Code terminal, from the project root, run `python -m pip install -r dashboard/requirements.txt`, then `python -m streamlit run dashboard/app.py`. Open the local URL printed in the terminal (normally `http://127.0.0.1:8501`). Press Ctrl+C in the terminal to stop the dashboard.

**Captain review** shows the saved operational ranking and counts recorded during the alert week. Search by captain ID or select a queue row to see which alerts occurred and how often. **Model & data** shows the official mean five-fold ratings, evaluation outcomes and ROC curves, alert feature correlations, and the saved training weights. The app reads the final project files directly and checks the displayed inputs against the saved run.

## Training and evaluation

The training examples pair March 1–7 alerts with March 8–14 reports, and March 8–14 alerts with March 15–21 reports. The final evaluation pairs March 18–24 alerts with March 25–31 reports. Duplicate-annotated reports and reports without a matching March alert captain are excluded.

The ANN uses all ten alert types in four groups. Repeated alerts in a day are discounted, and frequently occurring types receive lower weights estimated from training records. Its nine numeric inputs cover the four weighted alert rates, the share of observed days on which each group appeared, and the number of observed days. The network has 64 PReLU, 32 ReLU, and 16 ReLU hidden units and one sigmoid output.

Five captain-disjoint folds each score a different group using a model fitted only to earlier weeks from the other groups. The command reports the arithmetic mean of the five within-fold ROC-AUCs and the mean of the five accuracies at a fixed 0.5 cutoff. This previously examined March week contains eight incident captains, so the estimates are preliminary.

## Outputs

`outputs/final/results.json` contains the official ratings. `evaluation_predictions.csv` records each captain's fold, score, and observed outcome for checking those ratings.

`march_25_31_review_list.csv` ranks captains using one model fitted on all eligible earlier training profiles. Its scores can differ from the evaluation scores. The saved model is `weighted_ann.pt`, and `alert_weights.csv` contains its ten frequency-based alert weights.
