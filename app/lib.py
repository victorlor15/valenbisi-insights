"""Shared helpers for the ValenBisi Insights Streamlit app.

Centralises data loading (cached), the live Open-Data API call (with a graceful
fallback to historical typical values) and a few small UI/utility helpers.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Valencia Open Data - real-time Valenbisi availability (same schema as the historical CSVs)
LIVE_DATASET = "valenbisi-disponibilitat-valenbisi-dsiponibilidad"
LIVE_ENDPOINTS = [
    f"https://valencia.opendatasoft.com/api/explore/v2.1/catalog/datasets/{LIVE_DATASET}/exports/json",
    f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset={LIVE_DATASET}&rows=1000",
]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------- cached artifact loaders
@st.cache_data(show_spinner=False)
def load_station_meta() -> pd.DataFrame:
    return pd.read_parquet(PROC / "station_meta.parquet")


@st.cache_data(show_spinner=False)
def load_hourly_profiles() -> pd.DataFrame:
    return pd.read_parquet(PROC / "hourly_profiles.parquet")


@st.cache_data(show_spinner=False)
def load_training_sample() -> pd.DataFrame:
    return pd.read_parquet(PROC / "training_sample.parquet")


@st.cache_resource(show_spinner=False)
def load_model() -> dict | None:
    import joblib

    path = MODELS / "forecast_gbr.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def artifacts_ready() -> bool:
    return (PROC / "station_meta.parquet").exists()


# ----------------------------------------------------------------------------- live API
def _normalise_records(records: list[dict]) -> pd.DataFrame:
    """Map heterogeneous live-API field names to a common schema."""
    out = []
    for rec in records:
        f = rec.get("fields", rec)  # v1 wraps in 'fields', exports/json is flat
        geo = f.get("geo_point_2d")
        lat = lon = None
        if isinstance(geo, dict):
            lat, lon = geo.get("lat"), geo.get("lon")
        elif isinstance(geo, (list, tuple)) and len(geo) == 2:
            lat, lon = geo[0], geo[1]
        elif isinstance(geo, str) and "," in geo:
            lat, lon = (float(x) for x in geo.split(",")[:2])
        out.append({
            "station": f.get("number_", f.get("number")),
            "name": f.get("name"),
            "address": f.get("address"),
            "available": f.get("available"),
            "free": f.get("free"),
            "total": f.get("total"),
            "open": f.get("open"),
            "lat": lat,
            "lon": lon,
        })
    df = pd.DataFrame(out).dropna(subset=["station", "available", "total"])
    df["station"] = pd.to_numeric(df["station"], errors="coerce").astype("Int64")
    for c in ["available", "free", "total", "lat", "lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["station", "lat", "lon"])


@st.cache_data(ttl=300, show_spinner="Fetching live station status…")
def fetch_live() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label). Falls back to historical typical values."""
    for url in LIVE_ENDPOINTS:
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "ValenBisiInsights/1.0"})
            if r.status_code != 200:
                continue
            data = r.json()
            records = data.get("records", data) if isinstance(data, dict) else data
            df = _normalise_records(records)
            if len(df) > 50:
                df["occ_ratio"] = (df["available"] / df["total"]).clip(0, 1)
                return df, "live"
        except (requests.RequestException, ValueError):
            continue
    return _historical_now(), "historical"


def _historical_now() -> pd.DataFrame:
    """Approximate 'current' status from historical hourly profiles for now()."""
    meta = load_station_meta()
    prof = load_hourly_profiles()
    now = pd.Timestamp.now()
    snap = prof[(prof["dow"] == now.dayofweek) & (prof["hour_of_day"] == now.hour)]
    df = meta.merge(snap[["station", "available", "free", "occ_ratio"]],
                    on="station", how="inner")
    df["available"] = df["available"].round()
    df["free"] = df["free"].round()
    return df


# ----------------------------------------------------------------------------- utilities
def occupancy_color(ratio: float) -> list[int]:
    """Red (empty) -> yellow -> green (full) RGBA for pydeck."""
    if ratio is None or math.isnan(ratio):
        return [150, 150, 150, 160]
    r = max(0.0, min(1.0, ratio))
    if r < 0.5:
        red, green = 230, int(230 * (r / 0.5))
    else:
        red, green = int(230 * (1 - (r - 0.5) / 0.5)), 200
    return [red, green, 60, 180]


def haversine(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lon points (vectorised)."""
    import numpy as np

    r = 6_371_000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
