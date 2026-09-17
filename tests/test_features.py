"""Unit tests for src/features.py, focused on the numeric edge cases the
raw blend data can actually produce: rows where all property values in a
group are 0 or identical, a blend dominated by a single component, and
missing values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    EPS,
    N_COMPONENTS,
    N_PROPERTIES,
    add_aggregate_features,
    add_cross_component_stats,
    add_diff_features,
    add_diversity_features,
    add_entropy_features,
    add_rank_features,
    add_spread_features,
    add_weighted_interaction_features,
    add_zscore_features,
    build_features,
    property_columns,
    property_group_cols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_df(fractions=(0.2, 0.2, 0.2, 0.2, 0.2), overrides: dict[int, list[float]] | None = None) -> pd.DataFrame:
    """Build a single-row raw DataFrame.

    ``overrides`` maps a property index (1-10) to a list of 5 component
    values; any property not overridden defaults to [1, 2, 3, 4, 5], a
    fixed, non-degenerate group.
    """
    overrides = overrides or {}
    data: dict[str, list[float]] = {}
    for i in range(1, N_COMPONENTS + 1):
        data[f"Component{i}_fraction"] = [fractions[i - 1]]
    for j in range(1, N_PROPERTIES + 1):
        values = overrides.get(j, [1.0, 2.0, 3.0, 4.0, 5.0])
        for i in range(1, N_COMPONENTS + 1):
            data[f"Component{i}_Property{j}"] = [values[i - 1]]
    return pd.DataFrame(data)


# --- column helpers ---------------------------------------------------


def test_property_columns_counts_all_fifty():
    df = make_df()
    assert len(property_columns(df)) == N_COMPONENTS * N_PROPERTIES


def test_property_group_cols_names_five_components():
    cols = property_group_cols(3)
    assert cols == [f"Component{i}_Property3" for i in range(1, 6)]


# --- add_aggregate_features ---------------------------------------------


def test_add_aggregate_features_matches_manual_stats():
    df = make_df()
    out = add_aggregate_features(df)
    cols = property_columns(df)
    vals = out.loc[0, cols].to_numpy(dtype=float)
    assert out.loc[0, "prop_mean"] == pytest.approx(vals.mean())
    assert out.loc[0, "prop_min"] == pytest.approx(vals.min())
    assert out.loc[0, "prop_max"] == pytest.approx(vals.max())
    assert out.loc[0, "prop_range"] == pytest.approx(vals.max() - vals.min())


def test_add_aggregate_features_all_equal_values_zero_std_and_range():
    overrides = {j: [7.0] * 5 for j in range(1, N_PROPERTIES + 1)}
    df = make_df(overrides=overrides)
    out = add_aggregate_features(df)
    assert out.loc[0, "prop_std"] == pytest.approx(0.0)
    assert out.loc[0, "prop_range"] == pytest.approx(0.0)
    assert out.loc[0, "prop_mean"] == pytest.approx(7.0)


# --- add_cross_component_stats -------------------------------------------


def test_add_cross_component_stats_matches_manual_stats():
    df = make_df(overrides={5: [10.0, -2.0, 3.0, 0.0, 4.5]})
    out = add_cross_component_stats(df)
    vals = np.array([10.0, -2.0, 3.0, 0.0, 4.5])
    assert out.loc[0, "Property5_mean"] == pytest.approx(vals.mean())
    assert out.loc[0, "Property5_min"] == pytest.approx(vals.min())
    assert out.loc[0, "Property5_max"] == pytest.approx(vals.max())
    assert out.loc[0, "Property5_range"] == pytest.approx(vals.max() - vals.min())


def test_add_cross_component_stats_equal_group_zero_spread():
    df = make_df(overrides={2: [1.0, 1.0, 1.0, 1.0, 1.0]})
    out = add_cross_component_stats(df)
    assert out.loc[0, "Property2_std"] == pytest.approx(0.0)
    assert out.loc[0, "Property2_range"] == pytest.approx(0.0)


# --- add_weighted_interaction_features -----------------------------------


def test_weighted_features_single_dominant_component():
    df = make_df(fractions=(1.0, 0.0, 0.0, 0.0, 0.0), overrides={1: [9.0, 2.0, 3.0, 4.0, 5.0]})
    out = add_weighted_interaction_features(df)
    assert out.loc[0, "Weighted_Property1"] == pytest.approx(9.0)


def test_weighted_features_all_zero_fractions_gives_zero():
    df = make_df(fractions=(0.0, 0.0, 0.0, 0.0, 0.0))
    out = add_weighted_interaction_features(df)
    for j in range(1, N_PROPERTIES + 1):
        assert out.loc[0, f"Weighted_Property{j}"] == pytest.approx(0.0)


def test_weighted_features_component_columns_equal_fraction_times_property():
    df = make_df(fractions=(0.3, 0.7, 0.0, 0.0, 0.0), overrides={4: [2.0, 5.0, 1.0, 1.0, 1.0]})
    out = add_weighted_interaction_features(df)
    assert out.loc[0, "C1_P4_weighted"] == pytest.approx(0.3 * 2.0)
    assert out.loc[0, "C2_P4_weighted"] == pytest.approx(0.7 * 5.0)


# --- add_diversity_features ------------------------------------------------


def test_diversity_matches_max_minus_min():
    df = make_df(overrides={6: [1.0, 8.0, 3.0, -2.0, 0.0]})
    out = add_diversity_features(df)
    assert out.loc[0, "Property6_diversity"] == pytest.approx(8.0 - (-2.0))


def test_diversity_zero_when_group_equal():
    df = make_df(overrides={6: [4.0] * 5})
    out = add_diversity_features(df)
    assert out.loc[0, "Property6_diversity"] == pytest.approx(0.0)


# --- add_rank_features -------------------------------------------------


def test_rank_features_ties_all_rank_one():
    df = make_df(overrides={7: [3.0, 3.0, 3.0, 3.0, 3.0]})
    out = add_rank_features(df)
    for i in range(1, N_COMPONENTS + 1):
        assert out.loc[0, f"Component{i}_Property7_rank"] == 1.0


def test_rank_features_distinct_values_ordered():
    df = make_df(overrides={7: [30.0, 10.0, 50.0, 20.0, 40.0]})
    out = add_rank_features(df)
    # Component2 has the smallest value (10) -> rank 1; Component3 largest (50) -> rank 5
    assert out.loc[0, "Component2_Property7_rank"] == 1.0
    assert out.loc[0, "Component3_Property7_rank"] == 5.0


# --- add_entropy_features ------------------------------------------------


def test_entropy_all_zero_row_is_zero_not_nan():
    df = make_df(overrides={8: [0.0, 0.0, 0.0, 0.0, 0.0]})
    out = add_entropy_features(df)
    entropy_val = out.loc[0, "Property8_entropy"]
    assert entropy_val == pytest.approx(0.0)
    assert not np.isnan(entropy_val)


def test_entropy_uniform_distribution_is_max_entropy():
    df = make_df(overrides={8: [2.0, 2.0, 2.0, 2.0, 2.0]})
    out = add_entropy_features(df)
    assert out.loc[0, "Property8_entropy"] == pytest.approx(np.log(N_COMPONENTS))


def test_entropy_uses_absolute_value_sign_invariant():
    df_pos = make_df(overrides={8: [2.0, 2.0, 2.0, 2.0, 2.0]})
    df_mixed_sign = make_df(overrides={8: [-2.0, 2.0, -2.0, 2.0, -2.0]})
    entropy_pos = add_entropy_features(df_pos).loc[0, "Property8_entropy"]
    entropy_mixed = add_entropy_features(df_mixed_sign).loc[0, "Property8_entropy"]
    assert entropy_pos == pytest.approx(entropy_mixed)


# --- add_spread_features -------------------------------------------------


def test_spread_cv_zero_when_group_equal():
    df = make_df(overrides={9: [5.0] * 5})
    out = add_spread_features(df)
    assert out.loc[0, "Property9_cv"] == pytest.approx(0.0, abs=1e-9)


def test_spread_cv_no_zero_division_when_mean_is_zero():
    df = make_df(overrides={9: [-3.0, 3.0, -1.0, 1.0, 0.0]})
    out = add_spread_features(df)
    cv = out.loc[0, "Property9_cv"]
    assert np.isfinite(cv)


# --- add_zscore_features -------------------------------------------------


def test_zscore_matches_manual_computation():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    df = make_df(overrides={1: list(values)})
    out = add_zscore_features(df)
    mean, std = values.mean(), values.std(ddof=1) + EPS
    expected = (values - mean) / std
    for i in range(N_COMPONENTS):
        assert out.loc[0, f"Component{i + 1}_Property1_zscore"] == pytest.approx(expected[i])


def test_zscore_equal_group_no_nan_or_inf():
    df = make_df(overrides={1: [4.0] * 5})
    out = add_zscore_features(df)
    for i in range(1, N_COMPONENTS + 1):
        z = out.loc[0, f"Component{i}_Property1_zscore"]
        assert np.isfinite(z)


# --- add_diff_features -----------------------------------------------------


def test_diff_features_matches_manual_computation():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    df = make_df(overrides={1: list(values)})
    out = add_diff_features(df)
    mean, mn, mx = values.mean(), values.min(), values.max()
    assert out.loc[0, "Component1_Property1_diff_mean"] == pytest.approx(1.0 - mean)
    assert out.loc[0, "Component1_Property1_diff_min"] == pytest.approx(1.0 - mn)
    assert out.loc[0, "Component1_Property1_diff_max"] == pytest.approx(1.0 - mx)
    assert out.loc[0, "Property1_sum"] == pytest.approx(values.sum())


# --- build_features (full pipeline) ---------------------------------------


def test_build_features_smoke_on_synthetic_row():
    df = make_df()
    out = build_features(df)
    assert len(out.columns) > len(df.columns)
    assert out.isnull().sum().sum() == 0


def test_build_features_handles_missing_value_without_raising():
    df = make_df()
    df.loc[0, "Component1_Property1"] = np.nan
    out = build_features(df)  # should not raise
    # pandas .mean()/.std() skip NaN by default, so group-level stats are
    # still computed from the other 4 components...
    assert not np.isnan(out.loc[0, "Property1_mean"])
    # ...but any feature that directly reads the missing cell stays NaN, and
    # unrelated property groups are completely unaffected.
    assert np.isnan(out.loc[0, "Component1_Property1_zscore"])
    assert not np.isnan(out.loc[0, "Property2_mean"])
    assert not np.isnan(out.loc[0, "Component1_Property2_zscore"])


def test_build_features_on_real_training_sample():
    train_path = REPO_ROOT / "dataset" / "train.csv"
    df = pd.read_csv(train_path, nrows=20)
    out = build_features(df)
    assert len(out) == 20
    assert len(out.columns) > len(df.columns)
