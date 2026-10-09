<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/kinentrix_lockup_dark.png">
    <img src="assets/kinentrix_lockup_light.png" alt="KINENTRIX" width="300">
  </picture>
</p>

<h3 align="center">KINENTRIX Biologics 0.90</h3>

<p align="center">
  Human pharmacokinetic (PK) prediction for antibody therapeutics, delivered as a local web application
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-0.90-0b1f4d">
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-1f6feb">
  <img alt="Streamlit" src="https://img.shields.io/badge/streamlit-1.63.0-ff4b4b">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-0078d4">
  <img alt="License" src="https://img.shields.io/badge/license-proprietary-555555">
</p>

---

## Overview

**KINENTRIX Biologics** predicts human **clearance (CL)** and **volume of distribution (V)**
for an antibody therapeutic from its molecular properties and preclinical (cynomolgus
monkey) PK data, then simulates the resulting human concentration–time profile for a chosen
dosing regimen.

> **Public release** — The database covers 441 antibody drugs; this release opens data for
> five of them (Pembrolizumab, Canakinumab, Dupilumab, Nivolumab, Avelumab).

**Purpose**

- Support first-in-human dose selection and candidate ranking for antibody therapeutics.
- Compare classical **allometric scaling** with **machine-learning** predictions side by
  side, so the user can see how much each method performs compared to clinical data for a given molecule.
- Translate predicted CL / V directly into clinically meaningful exposure metrics
  (AUC, Cmax, Ctrough, half-life) with a population variability band.

**At a glance**

| Item | Value |
|---|---|
| Application | Streamlit web app, runs locally on `http://localhost:8510` |
| Version | 0.90 |
| Drug database | 441 antibody drugs |
| Prediction methods | Allometric scaling and machine-learning models (hybrid and direct) |
| Validation | Repeated cross-validation, out-of-fold predictions for training drugs |
| Runtime only | No training at start-up — prebuilt models are loaded and used for prediction |

---

## Architecture and workflow

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│ 1. INPUTS          │     │ 2. PREDICTION      │     │ 3. SIMULATION      │     │ 4. OUTPUT          │
├────────────────────┤     ├────────────────────┤     ├────────────────────┤     ├────────────────────┤
│ Molecule           │     │ Allometric scaling │     │ PK profile for the │     │ Predicted CL / V   │
│  isotype, category,│     │  (monkey CL / V)   │     │  regimen (IV / SC, │     │ Model comparison   │
│  MW, DAR, Kd ...   │ ──▶│ Machine learning   │ ──▶ │  dose, interval)   │──▶ │ PK profile chart   │
│ Monkey CL / V      │     │  (hybrid / direct) │     │ Population         │     │ AUC, Cmax, Ctrough │
│ Dosing regimen     │     │ Applicability check│     │ variability band   │     │ CSV / PNG export   │
│                    │     │                    │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └────────────────────┘     └────────────────────┘
```

1. **Inputs** — molecular descriptors, monkey CL / V and the dosing regimen.
2. **Prediction** — human CL and V are predicted by allometric scaling and by
   machine-learning models. The best-performing method per target is pre-selected (★)
   and can be changed by the user. An applicability check flags molecules outside the
   range of the training data.
3. **Simulation** — the predicted parameters drive a compartmental PK simulation of the
   regimen, with a population (inter-individual variability) band.
4. **Output** — predicted values, model comparison table, PK profile chart and
   downloadable CSV / PNG files.

### Prediction methods

| Method | Description |
|---|---|
| **Allometry** | Scales monkey CL / V to humans by body weight |
| **Hybrid** | Machine-learning correction applied on top of the allometric prediction |
| **Pure ML** | Machine-learning model that predicts human CL / V directly |

### Runtime flow (Existing Drugs tab)

```
app.py  load_all()                         ← once at app start (st.cache_resource)
  ├─ src/appdata.py   base()/model_idx()…  ← reads models/appdata.joblib
  ├─ src/predict.py   train(ds)            ← no training: loads betabeta_v1 + noseq_v1 bundles
  └─ src/predict_orig build_ood_index()    ← OOD reference set

Drug selected →
app.py  run_prediction()
  ├─ predict_orig.allometry_cl / allometry_v       ★ Allometry
  ├─ app.bundle_allometry()                        ★ training drugs use bundle baseline
  ├─ predict.build_features()                      ★ features
  └─ predict.apply_correction()                    ★ final-model prediction
app.py  target_predictions()
  ├─ training drug → predict.oof_model_predictions()  (out-of-fold, no memorization)
  └─ otherwise     → predict.model_predictions()
                       └─ predict._predict_one()    ★ Hybrid / Pure ML computed here
app.py  best_by_r2() → default (★) → apply_ood_fallback()
app.py  predict.flag() → traffic light → profile_figure() → src/pksim.simulate()
```

---

## Installation and usage (Windows)

### Quick start

1. Double-click **`run_app.vbs`**. It starts the app on **http://localhost:8510** and opens
   the browser.
2. On the first run, `.venv` does not exist yet, so it offers to run **`setup.cmd`**. This
   creates `.venv` in the app folder and installs the pinned packages from
   `requirements.txt` (one time, about 5–10 minutes, internet required).
   If Python is not installed, `setup.cmd` installs it for the current user with `winget`.

### What `run_app.vbs` does

1. Checks that the model files are present.
2. Looks for Python in `.\.venv\Scripts\python.exe`; if missing, offers to run `setup.cmd`.
3. Starts Streamlit on port 8510 in a minimized window (waits up to 60 s for the port).
4. Opens `http://localhost:8510/` in the default browser.

> The server listens on `0.0.0.0`, so other PCs on the same network can also connect.

### Deploy on Render

Render reads `render.yaml` (Blueprint): plan `1c-2g` (1 CPU / 2 GB) and
`KINENTRIX_DEMO_TOP5=1`. A password screen can be added by setting `KINENTRIX_PASSWORD`.

### Public release

This repository (`Kinentrix_Biologics_0.90`) is the public release (five drugs). The
application code is the same as the full version; the public data carries a `public` block
that switches on the Top-5 lock and the summary-based applicability check.
`tools/check_public_data.py` checks that the data files hold the five demo drugs only.

---

## Core capabilities

### 1. Existing Drugs — validation on 441 approved and clinical antibodies
- Drug information, monkey CL / V and predicted human CL / V for every drug in the database.
- **Out-of-fold predictions** for training drugs: each value comes from a model that never
  saw that drug, so the displayed accuracy is honest.
- **Traffic-light accuracy flags** against observed human values
  (CL: green < 2-fold, red > 3-fold · V: green < 1.5-fold, red > 2-fold).
- **Model comparison table** — 21 methods with cross-validated R², GMFE and % within 2-fold,
  including an "R² (same drugs)" column for a fair comparison with allometry.
- Simulated PK profile at the **approved regimen** (IV / SC switch) with observed Cmax /
  Ctrough overlaid.

### 2. New Drug — prediction for a new molecule
- Input form: molecular properties (isotype, category, MW, DAR, Kd …), monkey PK,
  in vitro data and the dosing regimen.
- Uses a dedicated **sequence-free model variant**, because new molecules usually lack
  sequence data.
- **Multi-method profile overlay** — several methods drawn on one chart.
- **OOD check**: if the molecule lies outside the training distribution, a warning is
  shown and Allometry is recommended.

### 3. Closest drugs
- Reference drugs ranked by **molecular similarity** and by **predicted PK similarity**,
  to put a new molecule in the context of known antibodies.

### 4. Automatic method selection
- The default (★) is the candidate with the highest cross-validated R² available for
  that drug; the user can switch to any other method.
- Hybrid models that become identical to their Pure ML twin (missing monkey data) are
  removed from the list automatically.

### 5. PK simulation engine
- Linear compartmental models, IV bolus / infusion and SC first-order absorption,
  multiple doses.
- Steady-state **AUCτ, Cmax, Ctrough** and terminal half-life.
- **Population variability band** and literature popPK band with body-weight covariate.

### 6. Export and display
- Profile and parameter tables (**CSV**) and figures (**PNG**).
- Light and dark display modes.

### Cross-validation summary

| Target | Method | n | R² | GMFE | Within 2-fold |
|---|---|---|---|---|---|
| CL | Allometry | 61 | 0.604 | 2.03× | 62% |
| CL | CatBoost (Hybrid) | 101 | 0.579 | 1.99× | 65% |
| CL | **XGBoost (Hybrid) ★ final** | 101 | 0.579 | 2.00× | 63% |
| CL | ExtraTrees (PureML) | 101 | 0.514 | 2.04× | 61% |
| V | Allometry | 113 | −0.375 | 1.52× | 81% |
| V | RandomForest (Hybrid) | 215 | 0.128 | 1.39× | 91% |
| V | **RandomForest (PureML) ★ final** | 215 | 0.138 | 1.38× | 91% |

---

## Contents

```
Kinentrix_Biologics_0.90/
├─ app.py                     Application entry point (Streamlit)
├─ config.py                  Application settings
├─ src/                       Prediction, simulation and export modules
├─ models/                    Prebuilt model and lookup files (no training data tables)
├─ assets/                    Logo and icon
├─ run_app.vbs, setup.cmd     Windows launcher and one-time setup
├─ requirements.txt           Python package versions
├─ render.yaml                Render deployment
├─ tools/check_public_data.py Checks the data files hold the five demo drugs only
└─ LICENSE                    License terms
```

---

## License

© 2026 APLUS Simulation. All rights reserved.

This software is proprietary. Its use is governed by the terms in [LICENSE](LICENSE).

