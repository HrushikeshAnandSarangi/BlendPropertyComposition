"""Feature engineering for the blend-property-prediction pipeline.

Every function takes a DataFrame containing the raw
``Component{1-5}_fraction`` and ``Component{1-5}_Property{1-10}`` columns
and returns a DataFrame with additional engineered columns appended. All
functions are pure (no in-place mutation of the input) and safe to call in
any order relative to each other, so each one is independently unit
testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy as _entropy

N_COMPONENTS = 5
N_PROPERTIES = 10
EPS = 1e-6

COMPONENT_RANGE = range(1, N_COMPONENTS + 1)
PROPERTY_RANGE = range(1, N_PROPERTIES + 1)


def property_columns(df: pd.DataFrame) -> list[str]:
    """All 50 Component{i}_Property{j} columns present in df."""
    return [c for c in df.columns if "_Property" in c]


def property_group_cols(j: int) -> list[str]:
    """The 5 component columns for a given property index j (1-10)."""
    return [f"Component{i}_Property{j}" for i in COMPONENT_RANGE]


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise stats across all 50 property values."""
    df = df.copy()
    cols = property_columns(df)
    df["prop_mean"] = df[cols].mean(axis=1)
    df["prop_std"] = df[cols].std(axis=1)
    df["prop_min"] = df[cols].min(axis=1)
    df["prop_max"] = df[cols].max(axis=1)
    df["prop_range"] = df["prop_max"] - df["prop_min"]
    df["prop_median"] = df[cols].median(axis=1)
    return df


def add_cross_component_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each property j, stats across the 5 components."""
    df = df.copy()
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        df[f"Property{j}_mean"] = df[cols].mean(axis=1)
        df[f"Property{j}_std"] = df[cols].std(axis=1)
        df[f"Property{j}_min"] = df[cols].min(axis=1)
        df[f"Property{j}_max"] = df[cols].max(axis=1)
        df[f"Property{j}_range"] = df[f"Property{j}_max"] - df[f"Property{j}_min"]
    return df


def add_weighted_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction-weighted property contributions, summed per property.

    ``Weighted_Property{j}`` approximates the linear mixing rule
    (sum of fraction_i * property_i_j across components) that blend
    properties often follow to first order.
    """
    df = df.copy()
    weighted_cols: dict[str, pd.Series] = {}
    for j in PROPERTY_RANGE:
        component_weighted = []
        for i in COMPONENT_RANGE:
            col = f"C{i}_P{j}_weighted"
            df[col] = df[f"Component{i}_fraction"] * df[f"Component{i}_Property{j}"]
            component_weighted.append(col)
        weighted_cols[f"Weighted_Property{j}"] = df[component_weighted].sum(axis=1)
    df = pd.concat([df, pd.DataFrame(weighted_cols, index=df.index)], axis=1)
    return df


def add_diversity_features(df: pd.DataFrame) -> pd.DataFrame:
    """How spread out (max - min) each property is across components."""
    df = df.copy()
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        df[f"Property{j}_diversity"] = df[cols].max(axis=1) - df[cols].min(axis=1)
    return df


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank (1..5) of each component's value within each property group."""
    df = df.copy()
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        rank_cols = [f"{c}_rank" for c in cols]
        df[rank_cols] = df[cols].rank(axis=1, method="min")
    return df


def _row_entropy(row: pd.Series) -> float:
    total = np.sum(np.abs(row.to_numpy()))
    if total < EPS:
        # All property values in this row are ~0 -> no information content.
        return 0.0
    probs = np.abs(row.to_numpy()) / total
    return float(_entropy(probs))


def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy of the (abs-normalized) component values per property.

    Guards against the zero-division that occurs when every component's
    value for a property is ~0 for a given row.
    """
    df = df.copy()
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        df[f"Property{j}_entropy"] = df[cols].apply(_row_entropy, axis=1)
    return df


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation per property (range/std provided elsewhere)."""
    df = df.copy()
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        mean = df[cols].mean(axis=1)
        std = df[cols].std(axis=1)
        df[f"Property{j}_cv"] = std / (mean.abs() + EPS)
    return df


def add_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row z-score of each component's value within its property group."""
    new_cols: dict[str, pd.Series] = {}
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        row_mean = df[cols].mean(axis=1)
        row_std = df[cols].std(axis=1) + EPS
        for col in cols:
            new_cols[f"{col}_zscore"] = (df[col] - row_mean) / row_std
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-component deviation from the group's mean/min/max, plus group sum."""
    new_cols: dict[str, pd.Series] = {}
    for j in PROPERTY_RANGE:
        cols = property_group_cols(j)
        row_mean = df[cols].mean(axis=1)
        row_min = df[cols].min(axis=1)
        row_max = df[cols].max(axis=1)
        for col in cols:
            new_cols[f"{col}_diff_mean"] = df[col] - row_mean
            new_cols[f"{col}_diff_min"] = df[col] - row_min
            new_cols[f"{col}_diff_max"] = df[col] - row_max
        new_cols[f"Property{j}_sum"] = df[cols].sum(axis=1)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


FEATURE_STEPS = (
    add_aggregate_features,
    add_cross_component_stats,
    add_weighted_interaction_features,
    add_diversity_features,
    add_rank_features,
    add_entropy_features,
    add_spread_features,
    add_zscore_features,
    add_diff_features,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full engineered-feature pipeline, in order."""
    for step in FEATURE_STEPS:
        df = step(df)
    return df
