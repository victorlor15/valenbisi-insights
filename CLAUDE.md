# ValenBisi Insights - EDM project

Interactive Streamlit app analysing and predicting **public bike (Valenbisi) availability
in Valencia**, addressing the urban problem of station imbalance ("no bike when I need one /
no dock when I want to park"). Built for the EDM course assignment.

**Working language of the app + code comments: ENGLISH** (the deliverable must be in English).
Conversation with the user is in Spanish.

## Evaluation criteria we optimise for
Originality/innovation · Difficulty · Quality of the application · Use of DS methods.

## Deliverables
- Link to the online app (Streamlit Community Cloud).
- Source code (this repo, pushed to GitHub).

## Data sources
1. **Historical** (for EDA + ML training): `github.com/ceferra/valenbici` - ~730 daily ZIPs
   (Dec 2022 → Dec 2024). Each ZIP = 96 CSVs (one snapshot every 15 min). ~276 stations.
   - Raw zip URL pattern: `https://github.com/ceferra/valenbici/raw/master/DD-MM-YYYY.zip`
   - CSV inside: `valenbici_DD-MM-YYYY_HH-MM-SS.csv`, separator `;`.
   - Columns: gid; name; number_; address; open; available; free; total; ticket;
     updated_at; globalid; created_user; created_date; last_edited_user; last_edited_date;
     geo_shape; geo_point_2d ("lat,lon").
2. **Live** (real-time view): Valencia Open Data API v2.1
   - `https://valencia.opendatasoft.com/api/explore/v2.1/catalog/datasets/`
     `valenbisi-disponibilitat-valenbisi-dsiponibilidad/records`
   - CSV export: `.../download/?format=csv` (separator `;`). Updates every ~10 min.

## Architecture / app modules
1. **Live** - real-time map (pydeck), nearest station with bike/dock.
2. **Patterns** - historical EDA: occupancy heatmaps by hour x day-of-week, top/bottom stations.
3. **Station types** - KMeans clustering of stations by hourly availability profile -> typologies.
4. **Forecast** - Gradient Boosting predicts available bikes by station/hour/dow/month
   -> rebalancing recommendation (P(empty)/P(full)).

## Data strategy (keep deployed app lightweight)
- Offline pipeline (`scripts/`) downloads a SAMPLE of zips (stratified across the 2 years),
  aggregates to compact artifacts committed to repo:
  - `data/processed/station_meta.parquet` - station id, name, lat, lon, total docks.
  - `data/processed/hourly_profiles.parquet` - mean available/free by (station, dow, hour).
  - `data/processed/training_sample.parquet` - rows for ML.
  - `models/forecast_gbr.joblib` - trained model.
- The app loads these artifacts (fast); the Live tab hits the API at runtime.
- Heavy raw data stays out of git (see .gitignore). Only compact processed artifacts committed.

## Environment
- Local venv: `.venv` (Python 3.11.9). Streamlit Cloud will use `requirements.txt` (target 3.11).
- NOTE: agent's Bash tool has NO internet (sandbox proxy). The **PowerShell tool HAS internet** -
  use it for downloads/API checks.

## Commands
- Activate venv: `.\.venv\Scripts\Activate.ps1`
- Run app: `.\.venv\Scripts\streamlit.exe run app\Home.py`
- Build data sample: `.\.venv\Scripts\python.exe scripts\build_dataset.py`
- Train model: `.\.venv\Scripts\python.exe scripts\train_model.py`

## Progress
- [x] Topic + framework decided (Valenbisi + Streamlit)
- [x] Data sources identified & schema confirmed
- [x] Project scaffolding + venv
- [x] requirements + install
- [x] Data ingestion/preprocessing pipeline (104 days sampled -> parquet artifacts)
- [x] EDA + processed artifacts (station_meta, hourly_profiles, training_sample)
- [x] Clustering (KMeans page, silhouette)
- [x] Forecast model (HistGradientBoosting; MAE 2.93 vs 4.47 baseline, R2 0.66)
- [x] Streamlit app (4 modules) - Home + Live/Patterns/Types/Forecast
- [x] Local test (smoke_test.py: all pages OK; server health 200)
- [ ] GitHub repo + Streamlit Cloud deploy  <-- NEXT (needs user's GitHub)

## Key gotchas (resolved)
- HistGradientBoosting caps categorical cardinality at 255; 276 stations -> dropped
  categorical station, rely on unique lat/lon + capacity instead.
- Some daily CSVs ship a partial/different header -> read_day skips them (REQUIRED_COLS).
- NEVER use PowerShell Set-Content to edit .py here: it double-encodes UTF-8 emojis
  (mojibake). Use the Write/Edit tools instead.
- `use_container_width` deprecated -> use `width="stretch"`.
