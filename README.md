# Swinash
Hackathon at Melb Uni, DSCUBE 05/2026, Kaggle Stock Prediction 

## Local setup with uv

This repo uses `uv` so the practice environment does not depend on your global Python packages.

Install uv if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart PowerShell, or add uv to your current shell:

```powershell
$env:Path = "C:\Users\gbspi\.local\bin;$env:Path"
```

Create the isolated environment using your existing Python 3.10 interpreter:

```powershell
uv sync --python "C:\Users\gbspi\AppData\Local\Programs\Python\Python310\python.exe" --cache-dir .uv-cache --link-mode=copy
```

Run Phase 1:

```powershell
uv run --cache-dir .uv-cache python .\phase1_data_recon.py
```

This avoids pandas/numpy binary mismatch errors from the system Python packages because `uv` installs clean project-local dependencies into `.venv`.

## Practice workflow

Run the scripts in this order:

```powershell
uv run --cache-dir .uv-cache python .\phase1_data_recon.py
```

```powershell
uv run --cache-dir .uv-cache python .\phase1_5_clean_data.py
```

```powershell
uv run --cache-dir .uv-cache python .\phase2_visual_eda.py
```

```powershell
uv run --cache-dir .uv-cache python .\phase2_5_prepare_features.py
```

```powershell
uv run --cache-dir .uv-cache python .\phase3_xgboost_baseline.py
```

```powershell
uv run --cache-dir .uv-cache python .\score_submission.py
```

Current phases:

- Phase 1: inspect raw data and identify messiness.
- Phase 1.5: clean obvious issues and save cleaned files.
- Phase 2: visual EDA, correlations, mutual information, and outlier summaries.
- Phase 2.5: prepare feature files for modeling.
- Phase 3: train a simple XGBoost baseline and create a submission file.
- Score submission: compare the mock submission against `answer_key.csv`.
