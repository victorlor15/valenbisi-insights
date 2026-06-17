# 🚲 ValenBisi Insights

An interactive **Streamlit** application that analyses and **predicts public-bike
(Valenbisi) availability in Valencia**, tackling the everyday city problem of *station
imbalance* - empty stations when you need a bike, full stations when you need to park.

Built for the **EDM** course assignment (interactive data-science app on open city data).

## ✨ What it does

| Page | Description | Data-science method |
|------|-------------|---------------------|
| 🔴 **Live map** | Real-time status of every station; finds the nearest station with a free bike or dock | Live Open-Data API + geospatial nearest-neighbour (haversine) |
| 📊 **Patterns** | Citywide and per-station occupancy by hour × weekday; most imbalanced stations | Exploratory data analysis & aggregation |
| 🧩 **Station types** | Groups stations into typologies (residential / business / leisure …) | **K-Means clustering** of 24-hour occupancy profiles + silhouette score |
| 🔮 **Forecast** | Predicts bikes available at any station, hour and weekday → rebalancing advice | **Gradient Boosting** regression (`HistGradientBoostingRegressor`) |

## 📊 Data

- **Historical** (EDA + model training): [`github.com/ceferra/valenbici`](https://github.com/ceferra/valenbici)
  - daily ZIPs (Dec 2022 → Dec 2024), one snapshot every 15 minutes, ~276 stations.
- **Live**: [Valencia Open Data](https://valencia.opendatasoft.com) real-time Valenbisi
  availability dataset (refreshed ~every 10 minutes).

The offline pipeline downloads a **stratified sample** of days across the two years,
aggregates the snapshots to hourly resolution and writes compact artifacts that the app
loads instantly. Only these small artifacts (and the trained model) are committed - the
raw ZIPs stay local (`data/raw/`, git-ignored).

## 🗂️ Project structure

```
app/
  Home.py                 # entry point
  lib.py                  # shared data loading, live API, helpers
  pages/                  # the four interactive pages
scripts/
  build_dataset.py        # download sample + build processed parquet artifacts
  train_model.py          # train & evaluate the forecasting model
data/processed/           # committed compact artifacts (parquet)
models/                   # committed trained model + metrics
requirements.txt
```

## ▶️ Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# (optional) rebuild data + model from scratch - needs internet:
python scripts/build_dataset.py
python scripts/train_model.py

streamlit run app/Home.py
```

The repository already ships the processed artifacts and the trained model, so the app
runs out of the box without rebuilding.

## ☁️ Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) create an app pointing to
   `app/Home.py` on the `main` branch.
3. Streamlit installs `requirements.txt` automatically and serves a public URL.

> The **Live map** uses the real-time API when internet access is available and otherwise
> falls back to the typical status for the current hour estimated from historical data, so
> the app is always functional.

## 🔬 Methodology notes

- **Clustering** uses each station's mean weekday occupancy curve (24 features). K is
  user-selectable and the silhouette score is reported for cluster quality.
- **Forecasting** treats the station id as a native categorical feature and learns the
  non-linear hour × weekday × station interactions that drive availability. Reported with
  MAE / RMSE / R² against a naïve per-station-average baseline on a held-out test set.

## 📦 Reproducibility

Everything is reproducible from the two scripts. Sampling parameters (date range, step,
snapshots per day) are configurable at the top of `scripts/build_dataset.py`.
