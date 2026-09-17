"""Minimal Streamlit interface for the blend-property model.

Run after model.ipynb has produced models/blend_model.joblib (+ scaler,
feature_names, target_cols):

    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import N_COMPONENTS, N_PROPERTIES, build_features
from src.modeling import load_model

MODELS_DIR = Path("models")
TEST_CSV = Path("dataset/test.csv")

st.set_page_config(page_title="Blend Property Predictor", layout="wide")
st.title("Blend Property Predictor")
st.caption(
    "Enter a blend's component fractions and component properties to predict "
    "the resulting BlendProperty1-10 values — useful for checking a formulation "
    "against spec before actually mixing it."
)


@st.cache_resource
def load_artifacts():
    model = load_model(MODELS_DIR / "blend_model.joblib")
    scaler = load_model(MODELS_DIR / "scaler.joblib")
    feature_names = load_model(MODELS_DIR / "feature_names.joblib")
    target_cols = load_model(MODELS_DIR / "target_cols.joblib")
    return model, scaler, feature_names, target_cols


@st.cache_data
def load_examples():
    return pd.read_csv(TEST_CSV)


missing = [
    p for p in ("blend_model.joblib", "scaler.joblib", "feature_names.joblib", "target_cols.joblib")
    if not (MODELS_DIR / p).exists()
]
if missing:
    st.error(
        "Missing model artifact(s): " + ", ".join(missing) + ". "
        "Run model.ipynb end-to-end first to train and save the model."
    )
    st.stop()

model, scaler, feature_names, target_cols = load_artifacts()
examples = load_examples()

st.sidebar.header("Load an example blend")
example_id = st.sidebar.selectbox("Test set ID", ["(none)"] + examples["ID"].astype(str).tolist())
example_row = None
if example_id != "(none)":
    example_row = examples.loc[examples["ID"].astype(str) == example_id].iloc[0]

st.subheader("Component fractions")
fraction_cols = st.columns(N_COMPONENTS)
fractions = []
for i, col in enumerate(fraction_cols, start=1):
    default = float(example_row[f"Component{i}_fraction"]) if example_row is not None else 0.2
    fractions.append(col.number_input(f"Component{i}_fraction", min_value=0.0, max_value=1.0, value=default, step=0.01))

fraction_sum = sum(fractions)
if abs(fraction_sum - 1.0) > 1e-6:
    st.warning(f"Fractions sum to {fraction_sum:.3f}, not 1.0. Predictions may be less reliable off-distribution.")

st.subheader("Component properties")
property_data = {}
for i in range(1, N_COMPONENTS + 1):
    row_defaults = [
        float(example_row[f"Component{i}_Property{j}"]) if example_row is not None else 0.0
        for j in range(1, N_PROPERTIES + 1)
    ]
    property_data[f"Component{i}"] = row_defaults

property_df = pd.DataFrame(
    property_data, index=[f"Property{j}" for j in range(1, N_PROPERTIES + 1)]
).T
edited = st.data_editor(property_df, width="stretch")

if st.button("Predict blend properties", type="primary"):
    raw = {}
    for i in range(1, N_COMPONENTS + 1):
        raw[f"Component{i}_fraction"] = [fractions[i - 1]]
        for j in range(1, N_PROPERTIES + 1):
            raw[f"Component{i}_Property{j}"] = [edited.loc[f"Component{i}", f"Property{j}"]]

    input_df = pd.DataFrame(raw)
    engineered = build_features(input_df)
    X = engineered.reindex(columns=feature_names, fill_value=0.0)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)[0]
    result = pd.DataFrame({"BlendProperty": target_cols, "Predicted value": preds})
    st.subheader("Predicted blend properties")
    st.dataframe(result, width="stretch", hide_index=True)
