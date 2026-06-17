"""Patterns - exploratory analysis of how availability varies by hour and weekday."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import lib

st.set_page_config(page_title="Patterns · ValenBisi", page_icon="📊", layout="wide")
st.title("📊 Usage patterns")

if not lib.artifacts_ready():
    st.error("Run the data pipeline first (see Home page).")
    st.stop()

meta = lib.load_station_meta()
prof = lib.load_hourly_profiles()
prof = prof.merge(meta[["station", "name"]], on="station", how="left")

st.markdown(
    "Occupancy = bikes available / docks. Below, the **citywide rhythm** reveals the "
    "classic commuter signature; pick a station to see its own fingerprint."
)

# --------------------------------------------------------------- citywide heatmap
city = (prof.groupby(["dow", "hour_of_day"], as_index=False)
            .agg(occ_ratio=("occ_ratio", "mean")))
pivot = city.pivot(index="hour_of_day", columns="dow", values="occ_ratio")
pivot.columns = [lib.DOW_NAMES[c] for c in pivot.columns]

fig = px.imshow(
    pivot, color_continuous_scale="RdYlGn", aspect="auto",
    labels=dict(x="Day of week", y="Hour of day", color="Avg occupancy"),
    title="Citywide average occupancy (green = bikes available, red = empty)",
)
fig.update_yaxes(dtick=2)
st.plotly_chart(fig, width="stretch")

col1, col2 = st.columns(2)

# ---------------------------------------------------- weekday vs weekend curve
with col1:
    curve = (prof.assign(kind=prof["dow"].ge(5).map({False: "Weekday", True: "Weekend"}))
                 .groupby(["kind", "hour_of_day"], as_index=False)
                 .agg(available=("available", "mean")))
    f2 = px.line(curve, x="hour_of_day", y="available", color="kind",
                 labels={"hour_of_day": "Hour", "available": "Avg bikes available",
                         "kind": ""},
                 title="Weekday vs weekend availability")
    st.plotly_chart(f2, width="stretch")

# ---------------------------------------------------- most volatile stations
with col2:
    vol = (prof.groupby(["station", "name"])["occ_ratio"]
               .agg(lambda s: s.max() - s.min())
               .reset_index(name="swing")
               .sort_values("swing", ascending=False).head(12))
    f3 = px.bar(vol, x="swing", y="name", orientation="h",
                labels={"swing": "Daily occupancy swing (max - min)", "name": ""},
                title="Most imbalanced stations (largest daily swing)")
    f3.update_yaxes(autorange="reversed")
    st.plotly_chart(f3, width="stretch")

# ---------------------------------------------------- per-station fingerprint
st.subheader("Single-station fingerprint")
station_name = st.selectbox("Station", sorted(prof["name"].dropna().unique()))
sp = prof[prof["name"] == station_name]
spivot = sp.pivot(index="hour_of_day", columns="dow", values="available")
spivot.columns = [lib.DOW_NAMES[c] for c in spivot.columns]
f4 = go.Figure(go.Heatmap(z=spivot.values, x=list(spivot.columns), y=spivot.index,
                          colorscale="Blues", colorbar=dict(title="Bikes")))
f4.update_layout(title=f"Average bikes available - {station_name}",
                 xaxis_title="Day of week", yaxis_title="Hour of day", height=480)
st.plotly_chart(f4, width="stretch")
