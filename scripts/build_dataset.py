"""Build compact, committed data artifacts for the ValenBisi Insights app.

Downloads a *stratified sample* of daily ZIPs from the historical Valenbisi archive
(github.com/ceferra/valenbici), parses the 15-min snapshots, aggregates them to an
hourly resolution and writes three small parquet files the Streamlit app consumes:

  data/processed/station_meta.parquet   - one row per station (id, name, lat, lon, docks)
  data/processed/hourly_profiles.parquet- mean availability by (station, dow, hour)
  data/processed/training_sample.parquet- per (station, day, hour) rows used to train ML

Run:  python scripts/build_dataset.py
The raw ZIPs are cached under data/raw/zips (git-ignored) so re-runs are cheap.
"""
from __future__ import annotations

import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "zips"
OUT_DIR = ROOT / "data" / "processed"

ZIP_URL = "https://github.com/ceferra/valenbici/raw/master/{date}.zip"

# Sample the 2-year archive: from START to END, take one day every STEP_DAYS.
# Defaults give ~120 days spread across all seasons / weekdays -> rich but small.
START = date(2023, 1, 2)
END = date(2024, 12, 15)
STEP_DAYS = 6
# Within each day keep this many evenly-spaced snapshots before hourly aggregation
# (96 per day = every 15 min). None -> keep all.
MAX_SNAPSHOTS_PER_DAY = None

CSV_COLS = ["number_", "name", "address", "open", "available", "free", "total",
            "geo_point_2d"]
REQUIRED_COLS = {"number_", "available", "total", "geo_point_2d"}
# --------------------------------------------------------------------------- helpers


def sample_dates() -> list[date]:
    days, d = [], START
    while d <= END:
        days.append(d)
        d += timedelta(days=STEP_DAYS)
    return days


def download_zip(d: date) -> Path | None:
    """Download one daily zip into the cache; return its path (None if unavailable)."""
    fname = d.strftime("%d-%m-%Y") + ".zip"
    dest = RAW_DIR / fname
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = ZIP_URL.format(date=d.strftime("%d-%m-%Y"))
    try:
        r = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"  ! {fname}: request failed ({exc})")
        return None
    if r.status_code != 200 or not r.content:
        print(f"  ! {fname}: HTTP {r.status_code}")
        return None
    dest.write_bytes(r.content)
    return dest


def parse_point(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """geo_point_2d is 'lat,lon' -> (lat, lon) float series."""
    parts = series.str.split(",", expand=True)
    lat = pd.to_numeric(parts[0], errors="coerce")
    lon = pd.to_numeric(parts[1], errors="coerce")
    return lat, lon


def read_day(zip_path: Path) -> pd.DataFrame | None:
    """Read all snapshots in a daily zip into a long, hourly-aggregated frame."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        if MAX_SNAPSHOTS_PER_DAY and len(names) > MAX_SNAPSHOTS_PER_DAY:
            idx = np.linspace(0, len(names) - 1, MAX_SNAPSHOTS_PER_DAY).astype(int)
            names = [names[i] for i in idx]
        for name in names:
            # filename: valenbici_DD-MM-YYYY_HH-MM-SS.csv
            stamp = name.replace("valenbici_", "").replace(".csv", "")
            try:
                ts = pd.to_datetime(stamp, format="%d-%m-%Y_%H-%M-%S")
            except ValueError:
                continue
            try:
                with zf.open(name) as fh:
                    df = pd.read_csv(fh, sep=";", usecols=lambda c: c in CSV_COLS)
            except (ValueError, pd.errors.ParserError):
                continue
            # some days ship snapshots with a different/partial header - skip those
            if not REQUIRED_COLS.issubset(df.columns):
                continue
            df["datetime"] = ts
            rows.append(df)
    if not rows:
        return None

    day = pd.concat(rows, ignore_index=True)
    day["number_"] = pd.to_numeric(day["number_"], errors="coerce")
    day = day.dropna(subset=["number_", "available", "total"])
    day = day[day["total"] > 0].copy()
    lat, lon = parse_point(day["geo_point_2d"])
    day["lat"], day["lon"] = lat, lon
    day["available"] = pd.to_numeric(day["available"], errors="coerce")
    day["free"] = pd.to_numeric(day["free"], errors="coerce")
    day["total"] = pd.to_numeric(day["total"], errors="coerce")
    day = day.dropna(subset=["available", "total"])

    # hourly aggregation per station (mean of the 4 quarter-hour snapshots)
    day["hour"] = day["datetime"].dt.floor("h")
    agg = (day.groupby(["number_", "hour"], as_index=False)
              .agg(available=("available", "mean"),
                   free=("free", "mean"),
                   total=("total", "median"),
                   name=("name", "first"),
                   address=("address", "first"),
                   lat=("lat", "median"),
                   lon=("lon", "median")))
    return agg


# --------------------------------------------------------------------------- main


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = sample_dates()
    print(f"Sampling {len(dates)} days from {START} to {END} (step {STEP_DAYS}d)")

    frames = []
    for i, d in enumerate(dates, 1):
        zp = download_zip(d)
        if zp is None:
            continue
        try:
            agg = read_day(zp)
        except (zipfile.BadZipFile, ValueError) as exc:
            print(f"  ! {d}: parse error ({exc})")
            continue
        if agg is not None:
            frames.append(agg)
        print(f"[{i}/{len(dates)}] {d}  ->  rows so far: "
              f"{sum(len(f) for f in frames):,}")

    if not frames:
        sys.exit("No data downloaded - check connectivity / URL pattern.")

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns={"number_": "station"})
    data["station"] = data["station"].astype(int)

    # time features
    data["dow"] = data["hour"].dt.dayofweek          # 0=Mon
    data["hour_of_day"] = data["hour"].dt.hour
    data["month"] = data["hour"].dt.month
    data["is_weekend"] = (data["dow"] >= 5).astype(int)
    data["occ_ratio"] = (data["available"] / data["total"]).clip(0, 1)

    # 1) station metadata (most frequent name/address, median coords) -----------------
    meta = (data.groupby("station")
                .agg(name=("name", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
                     address=("address", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
                     lat=("lat", "median"),
                     lon=("lon", "median"),
                     total=("total", "median"))
                .reset_index())
    meta["total"] = meta["total"].round().astype(int)
    meta.to_parquet(OUT_DIR / "station_meta.parquet", index=False)

    # 2) hourly profiles: typical availability by station x dow x hour ----------------
    profiles = (data.groupby(["station", "dow", "hour_of_day"], as_index=False)
                    .agg(available=("available", "mean"),
                         free=("free", "mean"),
                         occ_ratio=("occ_ratio", "mean"),
                         n=("available", "size")))
    profiles.to_parquet(OUT_DIR / "hourly_profiles.parquet", index=False)

    # 3) training sample ---------------------------------------------------------------
    train_cols = ["station", "hour_of_day", "dow", "month", "is_weekend",
                  "total", "available", "free", "occ_ratio", "lat", "lon"]
    train = data[train_cols].copy()
    train.to_parquet(OUT_DIR / "training_sample.parquet", index=False)

    print("\nDone.")
    print(f"  stations           : {meta.shape[0]}")
    print(f"  profile rows       : {profiles.shape[0]:,}")
    print(f"  training rows      : {train.shape[0]:,}")
    print(f"  date span (sampled): {data['hour'].min()}  ->  {data['hour'].max()}")


if __name__ == "__main__":
    main()
