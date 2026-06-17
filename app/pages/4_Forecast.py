"""Forecast - predict bikes available at a station for any hour / weekday."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

st.set_page_config(page_title="Forecast · ValenBisi", page_icon="🔮", layout="wide")
st.title("🔮 Availability forecast")

if not lib.artifacts_ready():
    st.error("Run the data pipeline first (see Home page).")
    st.stop()

payload = lib.load_model()
if payload is None:
    st.error("Model not found. Run `python scripts/train_model.py`.")
    st.stop()

model = payload["model"]
features = payload["features"]
meta = lib.load_station_meta()
prof = lib.load_hourly_profiles()

m = payload["metrics"]
st.markdown(
    f"A **Gradient Boosting** model trained on two years of data predicts how many bikes "
    f"will be available. Test-set error: **MAE {m['mae']:.2f} bikes** "
    f"(R² {m['r2']:.2f}), versus {m['baseline_mae']:.2f} for a naive per-station average."
)

c1, c2, c3 = st.columns(3)
with c1:
    station_name = st.selectbox("Station", sorted(meta["name"].dropna().unique()))
with c2:
    dow_name = st.selectbox("Day of week", lib.DOW_NAMES, index=0)
with c3:
    month = st.select_slider("Month", options=list(range(1, 13)),
                             value=pd.Timestamp.now().month)
dow = lib.DOW_NAMES.index(dow_name)
srow = meta[meta["name"] == station_name].iloc[0]
station_id = int(srow["station"])
total = int(srow["total"])


def predict_curve(dow, month, total, lat, lon):
    hours = np.arange(24)
    X = pd.DataFrame({
        "hour_of_day": hours,
        "dow": dow,
        "month": month,
        "is_weekend": int(dow >= 5),
        "total": total,
        "lat": lat,
        "lon": lon,
    })[features]
    pred = np.clip(model.predict(X), 0, total)
    return hours, pred


hours, pred = predict_curve(dow, month, total, srow["lat"], srow["lon"])

# historical typical for the same station/dow as a reference overlay
hist = (prof[(prof["station"] == station_id) & (prof["dow"] == dow)]
        .set_index("hour_of_day")["available"].reindex(hours))

fig = go.Figure()
fig.add_trace(go.Scatter(x=hours, y=pred, name="Model forecast", mode="lines+markers",
                         line=dict(color="#1f77b4", width=3)))
if hist.notna().any():
    fig.add_trace(go.Scatter(x=hours, y=hist.values, name="Historical average",
                             mode="lines", line=dict(color="#999", dash="dash")))
fig.add_hline(y=total, line=dict(color="green", dash="dot"),
              annotation_text="capacity (all docks full)")
fig.add_hline(y=1.5, line=dict(color="red", dash="dot"),
              annotation_text="risk of empty")
fig.update_layout(title=f"Predicted bikes available - {station_name} ({dow_name})",
                  xaxis_title="Hour of day", yaxis_title="Bikes available",
                  xaxis=dict(dtick=2), height=460)
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- rebalancing insight
pred_round = np.round(pred).astype(int)
empty_hours = hours[pred_round <= 1]
full_hours = hours[pred_round >= total - 1]

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🚲 Likely **no bikes**")
    st.write(", ".join(f"{h:02d}:00" for h in empty_hours) or "- none predicted -")
with col2:
    st.markdown("#### 🅿️ Likely **no free docks**")
    st.write(", ".join(f"{h:02d}:00" for h in full_hours) or "- none predicted -")

st.info(
    "**Operational reading:** stations short of bikes in the morning and full in the "
    "evening (or vice-versa) are rebalancing candidates - vans should refill them just "
    "before the predicted shortage window.",
    icon="🛠️",
)

with st.expander("Why a Gradient Boosting model?"):
    st.markdown(
        "Availability depends on **non-linear interactions** between the hour, the weekday "
        "and the location of the station (a campus station and a beach station peak at "
        "opposite times). `HistGradientBoostingRegressor` captures these interactions, "
        "identifies each station from its unique coordinates and capacity, and trains in "
        "seconds - a strong, compact choice for tabular data versus linear models or a "
        "heavier neural network."
    )
