"""Live map - real-time status of every Valenbisi station + nearest-station finder."""
from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

import lib

st.set_page_config(page_title="Live map · ValenBisi", page_icon="🔴", layout="wide")
st.title("🔴 Live station map")

if not lib.artifacts_ready():
    st.error("Run the data pipeline first (see Home page).")
    st.stop()

df, source = lib.fetch_live()
if source == "historical":
    st.warning(
        "The live Open-Data API could not be reached, so the map shows the **typical "
        "status for the current hour** estimated from historical data. On a machine with "
        "open internet access (e.g. Streamlit Cloud) the real-time feed is used.",
        icon="⚠️",
    )
else:
    st.success(f"Live feed loaded - {len(df)} stations. Cached for 5 minutes.", icon="✅")

# -------------------------------------------------------------------- summary metrics
empty = int((df["available"] <= 1).sum())
full = int((df["free"] <= 1).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations", f"{len(df)}")
c2.metric("Bikes available", f"{int(df['available'].sum()):,}")
c3.metric("Empty stations (<=1 bike)", empty)
c4.metric("Full stations (<=1 dock)", full)

# ------------------------------------------------------------------------ nearest finder
st.subheader("Find the nearest station")
left, right = st.columns([1, 2])
with left:
    need = st.radio("I need a…", ["Bike 🚲", "Free dock 🅿️"], horizontal=False)
    ref_name = st.selectbox("Near station", sorted(df["name"].dropna().unique()))
    ref = df[df["name"] == ref_name].iloc[0]
    want_bike = need.startswith("Bike")
    pool = df[df["available"] >= 1] if want_bike else df[df["free"] >= 1]
    pool = pool[pool["name"] != ref_name].copy()
    pool["dist_m"] = lib.haversine(ref["lat"], ref["lon"], pool["lat"], pool["lon"])
    nearest = pool.nsmallest(5, "dist_m")
    st.caption(f"Reference: **{ref_name}** - {int(ref['available'])} bikes / "
               f"{int(ref['free'])} free docks now.")
with right:
    show = nearest[["name", "available", "free", "dist_m"]].rename(
        columns={"name": "Station", "available": "Bikes", "free": "Free docks",
                 "dist_m": "Distance (m)"})
    show["Distance (m)"] = show["Distance (m)"].round().astype(int)
    st.dataframe(show, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- the map
df = df.copy()
df["color"] = df["occ_ratio"].apply(lib.occupancy_color)
df["tooltip"] = (df["name"].astype(str) + " - " + df["available"].astype(int).astype(str)
                 + " bikes / " + df["free"].astype(int).astype(str) + " docks")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=70,
    radius_min_pixels=4,
    radius_max_pixels=18,
    pickable=True,
)
highlight = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame([ref]),
    get_position="[lon, lat]",
    get_fill_color="[30, 90, 230, 230]",
    get_radius=120,
    radius_min_pixels=7,
    pickable=False,
)
view = pdk.ViewState(latitude=float(df["lat"].mean()),
                     longitude=float(df["lon"].mean()), zoom=12.2)
st.pydeck_chart(pdk.Deck(layers=[layer, highlight], initial_view_state=view,
                         map_style=None, tooltip={"text": "{tooltip}"}))
st.caption("🟢 plenty of bikes · 🟡 medium · 🔴 nearly empty · 🔵 your reference station")
