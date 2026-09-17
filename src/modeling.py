"""Model comparison, explainability, and persistence helpers.

Kept separate from feature engineering (`src/features.py`) so the notebook
stays thin: it calls into these functions and reports/plots the results
rather than re-implementing training/evaluation logic inline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def train_val_split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_candidate_models(random_state: int = 42) -> dict[str, MultiOutputRegressor]:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "CatBoost": MultiOutputRegressor(
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=random_state,
                verbose=False,
            )
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
                verbosity=0,
            )
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
                verbosity=-1,
            )
        ),
    }


def compare_models(X_tr, y_tr, X_val, y_val, target_names=None, random_state: int = 42):
    """Fit CatBoost, XGBoost, and LightGBM on the train split and score each
    on the held-out val split.

    Returns ``(summary_df, per_target_df, fitted_models)``:
      - ``summary_df``: one row per model, mean MAPE/MAE/RMSE across targets,
        sorted best (lowest MAPE) first.
      - ``per_target_df``: one row per (model, target) with the same metrics.
      - ``fitted_models``: dict of name -> fitted estimator, so the winner
        can be reused without retraining.
    """
    if target_names is None:
        target_names = (
            list(y_tr.columns) if hasattr(y_tr, "columns") else [f"target_{i}" for i in range(np.asarray(y_tr).shape[1])]
        )

    models = build_candidate_models(random_state=random_state)
    fitted_models: dict[str, MultiOutputRegressor] = {}
    summary_rows = []
    per_target_rows = []

    y_val_arr = np.asarray(y_val)

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        mapes, maes, rmses = [], [], []
        for idx, target in enumerate(target_names):
            true_col = y_val_arr[:, idx]
            pred_col = preds[:, idx]
            mape = mean_absolute_percentage_error(true_col, pred_col)
            mae = mean_absolute_error(true_col, pred_col)
            rmse = mean_squared_error(true_col, pred_col) ** 0.5
            mapes.append(mape)
            maes.append(mae)
            rmses.append(rmse)
            per_target_rows.append({"model": name, "target": target, "MAPE": mape, "MAE": mae, "RMSE": rmse})

        summary_rows.append(
            {
                "model": name,
                "mean_MAPE": float(np.mean(mapes)),
                "mean_MAE": float(np.mean(maes)),
                "mean_RMSE": float(np.mean(rmses)),
            }
        )
        fitted_models[name] = model

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_RMSE").reset_index(drop=True)
    per_target_df = pd.DataFrame(per_target_rows)
    return summary_df, per_target_df, fitted_models


def shap_summary(model: MultiOutputRegressor, X, feature_names, target_names, top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Top-N SHAP-driving features per target for a fitted MultiOutputRegressor.

    Each of CatBoost/XGBoost/LightGBM's per-target estimators is a tree
    model, so `shap.TreeExplainer` applies directly to `model.estimators_[i]`.
    Returns ``{target_name: DataFrame(feature, mean_abs_shap)}``, sorted by
    importance descending.
    """
    import shap

    X_arr = np.asarray(X)
    feature_names = np.array(feature_names)
    results: dict[str, pd.DataFrame] = {}

    for idx, target in enumerate(target_names):
        estimator = model.estimators_[idx]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_arr)
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        results[target] = pd.DataFrame(
            {"feature": feature_names[order], "mean_abs_shap": mean_abs[order]}
        ).reset_index(drop=True)

    return results


def save_model(model, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
