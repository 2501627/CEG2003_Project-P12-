# Captain Safety Lab — setup

The active model is **64 PReLU → 32 ReLU → 16 ReLU → 1 sigmoid**, with three hidden layers and learned alert-type weights. Its training-selected cutoff gives **ROC-AUC 0.601, accuracy 66.06%, incident detection 52.78%** on the saved historical tests. It uses class weighting without SMOTE. Weights stay fixed until retraining.

## Run in VS Code

Open `C:\Codex\CEG2003_Insturment-Data`, then **Terminal → New Terminal**. All required packages are already installed on this computer:

```powershell
& "C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe" -B -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open **http://127.0.0.1:8501**. Keep the terminal running; press **Ctrl+C** to stop. Restart an existing dashboard after an update. If the port is occupied, replace `8501` with `8502` in both the command and address.

This command works when PowerShell blocks scripts; no security-setting change is needed. The optional launcher is `./run_dashboard.ps1`, with `-Check` to verify readiness. F5 and **Dashboard: active model** in VS Code use this computer's installed Python.

## Install on another computer

Use Python 3.13 and run from the project root:

```powershell
py -3.13 -m venv .venv-dashboard
.\.venv-dashboard\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv-dashboard\Scripts\python.exe -B -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

The installation downloads Python packages. Data and the trained model are already local. Update the interpreter path in VS Code's launch/task settings on another computer, or use the terminal command directly.

## Project contents

- `data/raw/`, `data/processed/`, `data/audit/`: original Excel files, cleaned data and exclusions audit, all preserved.
- `data/model_inputs/`: numeric captain inputs and the original training/test groups.
- `models/learned_alert_ann/classifier.joblib`: active trained model.
- `config/learned_alert_ann.json`: active architecture and training settings.
- `outputs/learned_alert_ann/`: current test ratings, predictions and learned weights.
- `training/`: self-contained current ANN training code.
- `dashboard/`: inference, review queue, histories and model information.
- `scripts/prepare_dashboard.py`: refreshes display tables without retraining.
- `tests/test_dashboard_learned_ann.py`: current dashboard and model checks.

Obsolete models, experiment folders, checkpoints and rollback backups have been removed. The current project does not depend on them.

## Refresh, verify or train

Refresh dashboard tables from the same active model:

```powershell
python -B scripts/prepare_dashboard.py
```

Run the current checks:

```powershell
python -B -m unittest discover -s tests -p test_dashboard_learned_ann.py -v
```

Reproduce the current full-data training fit in a new output folder:

```powershell
python -B -m training.train
```

Repeat the original three rounds of five-fold testing, with training-only weight/epoch/cutoff selection:

```powershell
python -B -m training.train --evaluate --output outputs/evaluated_ann
```

Training outputs never overwrite or activate the current model automatically. Choose a fresh output directory for each run. The saved preprocessing and groups apply to this supplied dataset.

## Reading the dashboard

Priority /100 is a relative score percentile, not accident probability. Review capacity and model cutoff are separate selections. Historical queue scores use the full-data refit; ratings use held-out tests on only 36 incident captains. This repeatedly examined cohort provides exploratory internal evidence.

Learned type shares describe the input mixture. The two-input reference explanations describe score changes, not causal danger or fault. Days with alert records do not measure actual driving exposure. No eligible April report means no report in the workbook, not proof of no accident.
