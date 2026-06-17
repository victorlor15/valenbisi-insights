"""Station types - K-Means clustering of stations by their daily availability profile."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import lib

st.set_page_config(page_title="Station types · ValenBisi", page_icon="🧩", layout="wide")
st.title("🧩 Station typologies")
st.markdown(
    "Each station has a daily rhythm. We describe every station by its **24-hour weekday "
    "occupancy curve** and group similar curves with **K-Means**. The result is a small set "
    "of station *types* that explain the city's imbalance - and tell operators which "
    "stations behave alike."
)

if not lib.artifacts_ready():
    st.error("Run the data pipeline first (see Home page).")
    st.stop()

meta = lib.load_station_meta()
prof = lib.load_hourly_profiles()


@st.cache_data(show_spinner=False)
def feature_matrix() -> pd.DataFrame:
    """One row per station, 24 columns = mean weekday occupancy by hour."""
    weekday = prof[prof["dow"] < 5]
    mat = (weekday.groupby(["station", "hour_of_day"])["occ_ratio"].mean()
                  .unstack("hour_of_day"))
    return mat.dropna()


@st.cache_data(show_spinner=False)
def run_kmeans(k: int):
    mat = feature_matrix()
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(mat.values)
    sil = silhouette_score(mat.values, labels) if k > 1 else float("nan")
    return mat.index.to_numpy(), labels, km.cluster_centers_, mat.columns.to_numpy(), sil


k = st.slider("Number of station types (k)", 2, 6, 4)
stations, labels, centers, hours, sil = run_kmeans(k)
st.caption(f"Silhouette score: **{sil:.3f}** (higher = better separated clusters).")

clusters = pd.DataFrame({"station": stations, "cluster": labels}).merge(meta, on="station")

palette = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
           [148, 103, 189], [140, 86, 75]]
clusters["color"] = clusters["cluster"].apply(lambda c: palette[c % len(palette)] + [180])

left, right = st.columns([3, 2])

# ----------------------------------------------------------------- map of clusters
# NOTE: pydeck snapshots the dataframe when the Layer is built, so every column the
# tooltip references must exist *before* pdk.Layer(...) is created.
clusters["tooltip"] = clusters["name"] + " - type " + clusters["cluster"].astype(str)
with left:
    layer = pdk.Layer(
        "ScatterplotLayer", data=clusters, get_position="[lon, lat]",
        get_fill_color="color", get_radius=80, radius_min_pixels=4,
        radius_max_pixels=16, pickable=True)
    view = pdk.ViewState(latitude=float(clusters["lat"].mean()),
                         longitude=float(clusters["lon"].mean()), zoom=12.2)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style=None,
                             tooltip={"text": "{tooltip}"}))

# ----------------------------------------------------- cluster profile curves
with right:
    prof_df = pd.DataFrame(centers, columns=hours)
    prof_df.index.name = "cluster"
    long = prof_df.reset_index().melt("cluster", var_name="hour", value_name="occ")
    long["cluster"] = "Type " + long["cluster"].astype(str)
    fig = px.line(long, x="hour", y="occ", color="cluster",
                  labels={"hour": "Hour of day", "occ": "Avg occupancy"},
                  title="Typical weekday curve per type")
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------- auto interpretation
st.subheader("How to read the types")
rows = []
for c in range(k):
    center = centers[c]
    morning = center[6:11].mean()      # 06-10h
    evening = center[17:21].mean()     # 17-20h
    overall = center.mean()
    if morning - evening > 0.12:
        behaviour = "🌆 Residential - full at night, empties in the morning rush"
    elif evening - morning > 0.12:
        behaviour = "🏢 Business/transport hub - fills up during the working day"
    elif overall > 0.6:
        behaviour = "🟢 Usually well-stocked (bike-rich)"
    elif overall < 0.35:
        behaviour = "🔴 Usually scarce (bike-poor)"
    else:
        behaviour = "⚖️ Balanced / leisure - stable through the day"
    rows.append({"Type": c, "Stations": int((labels == c).sum()),
                 "Avg occupancy": round(float(overall), 2), "Behaviour": behaviour})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
