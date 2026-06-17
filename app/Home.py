"""ValenBisi Insights - main page.

Interactive analysis & forecasting of Valencia's public-bike (Valenbisi) availability.
Run locally:  streamlit run app/Home.py
"""
from __future__ import annotations

import streamlit as st

import lib

st.set_page_config(page_title="ValenBisi Insights", page_icon="🚲", layout="wide")

st.title("🚲 ValenBisi Insights")
st.markdown(
    "#### Understanding and predicting public-bike availability in Valencia\n"
    "Valencia's bike-share system suffers from **station imbalance**: in commuter areas "
    "stations empty out in the morning, while others fill up and leave no docks to park. "
    "This app analyses two years of historical data and a live feed to help riders and "
    "operators answer one question: **where will there be a bike (or a free dock) - and when?**"
)

if not lib.artifacts_ready():
    st.error(
        "Processed data not found. Run the data pipeline first:\n\n"
        "```\npython scripts/build_dataset.py\npython scripts/train_model.py\n```"
    )
    st.stop()

meta = lib.load_station_meta()
prof = lib.load_hourly_profiles()
model = lib.load_model()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations analysed", f"{meta['station'].nunique()}")
c2.metric("Total docks (capacity)", f"{int(meta['total'].sum()):,}")
c3.metric("Hourly profile rows", f"{len(prof):,}")
if model:
    c4.metric("Forecast model MAE", f"{model['metrics']['mae']:.2f} bikes")

st.divider()

st.markdown(
    """
### What you can explore

| Page | What it does | Data Science method |
|------|--------------|---------------------|
| 🔴 **Live map** | Real-time status of every station; find the nearest bike or free dock | Live API + geospatial nearest-neighbour |
| 📊 **Patterns** | How availability changes by hour and weekday across the city | Exploratory data analysis, aggregation |
| 🧩 **Station types** | Groups stations by their daily rhythm (residential / business / leisure) | **K-Means clustering** of hourly profiles |
| 🔮 **Forecast** | Predicts bikes available at any station, hour and weekday | **Gradient Boosting** regression |

Use the sidebar to navigate. ⬅️
"""
)

with st.expander("ℹ️ Data sources & methodology"):
    st.markdown(
        """
- **Historical data:** daily snapshots (every 15 min) of every Valenbisi station,
  Dec 2022 - Dec 2024, from the [`ceferra/valenbici`](https://github.com/ceferra/valenbici)
  archive - itself collected from Valencia's open-data portal.
- **Live data:** [Valencia Open Data](https://valencia.opendatasoft.com) real-time
  Valenbisi availability dataset (refreshed ~every 10 minutes).
- **Pipeline:** snapshots are aggregated to an hourly resolution; per-station typical
  profiles and a training table feed the clustering and forecasting models.
- All processing is reproducible via `scripts/build_dataset.py` and `scripts/train_model.py`.
"""
    )
    if model:
        m = model["metrics"]
        st.markdown(
            f"**Forecast model performance** - MAE **{m['mae']:.2f}** bikes "
            f"(naïve per-station baseline {m['baseline_mae']:.2f}), "
            f"RMSE {m['rmse']:.2f}, R² **{m['r2']:.2f}** on a held-out 20% test set."
        )
