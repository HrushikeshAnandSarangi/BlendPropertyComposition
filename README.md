# Blend Property Prediction

Predicting the resulting properties of a 5-component blend — e.g. a fuel
or chemical formulation — from each component's mixing fraction and its
own measured properties. This is a real formulation-and-quality-control
problem: given raw ingredient specs and a target mix ratio, predict the
blended product's properties before you actually blend it, so an
out-of-spec batch can be caught before it's produced rather than after.

Each row is a blend of up to 5 components. `Component{i}_fraction` is the
mixing ratio of component *i* (summing to 1 across the 5 components), and
`Component{i}_Property{1-10}` are 10 measured (standardized) properties of
that component. The targets, `BlendProperty1..10`, are the resulting
properties of the finished blend. The model comparison, evaluation
methodology, and SHAP interpretability results below are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Approach

1. **Feature engineering** (`src/features.py`, unit tested in
   `tests/test_features.py`) — beyond the raw fractions and properties:
   row-wise aggregate stats, per-property cross-component stats
   (mean/std/min/max/range across the 5 components), fraction-weighted
   linear-mixing-rule estimates, diversity/spread/coefficient-of-variation,
   component rank within each property group, Shannon entropy of each
   property group, per-component z-scores, and per-component deviations
   from the group mean/min/max — roughly 400 engineered features from the
   original 55 raw columns.
2. **Baseline evaluation** (`src/modeling.py`) — an 80/20 train/val split,
   then CatBoost, XGBoost, and LightGBM trained with matched
   hyperparameters and compared on held-out MAPE/MAE/RMSE, rather than
   fitting one model and assuming it's good.
3. **Explainability** — SHAP (`shap.TreeExplainer`) run on the winning
   model to identify which engineered features actually drive each of the
   10 `BlendProperty` predictions, not just how many features were built.
4. **Final model** — the winning architecture refit on the full training
   set, saved to `models/blend_model.joblib`, used to generate
   `submission.csv`.
5. **Interface** — a minimal Streamlit app (`app.py`) for interactively
   entering a blend and getting predicted properties back, using the saved
   model.

## Project layout

```
dataset/            train/test CSVs + sample submission
src/features.py     engineered feature pipeline (pure functions, unit tested)
src/modeling.py     train/val split, model comparison, SHAP, save/load
tests/               pytest suite for the feature pipeline
model.ipynb          end-to-end notebook: features -> comparison -> SHAP -> submission
app.py               Streamlit prediction interface
BENCHMARKS.md        model comparison results, SHAP findings, test coverage
```

## Running it

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# unit tests
pytest -q

# full pipeline (features -> model comparison -> SHAP -> submission.csv)
jupyter nbconvert --to notebook --execute --inplace model.ipynb

# interactive prediction UI (after the notebook has produced models/blend_model.joblib)
streamlit run app.py
```

Runs in a local virtual environment — the dependency set (CatBoost, XGBoost,
LightGBM, SHAP, Streamlit alongside the usual pandas/scikit-learn stack) is
heavy enough that it shouldn't be installed into a global Python
environment.
