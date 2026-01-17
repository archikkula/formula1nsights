## Formula 1nsights

Formula 1nsights is a full‑stack Formula 1 analytics and news platform that combines a React SPA frontend with a Flask‑based ML inference API. It continuously ingests race schedules and results, enriches them with circuit and weather data, trains gradient‑boosted ranking models, and surfaces:

- AI‑powered predictions for upcoming and historical races
- Live F1 news pulled from the official Formula 1 RSS feed
- Dashboards that compare model predictions to actual race outcomes

## Features

- **AI‑powered race predictions**
  - Two‑stage grid → finish pipeline using gradient‑boosted models.
  - Separate pre‑qualifying and post‑qualifying models for upcoming races.
  - Monte Carlo–style resampling for uncertainty estimates and confidence intervals.

- **Historical race analysis**
  - Re‑run models on past races to compare predicted vs actual results.
  - Per‑driver comparison for grid predictions vs qualifying outcomes.
  - Error inspection across seasons and circuits.

- **Live F1 news hub**
  - Scheduled RSS ingestion from the official Formula 1 feed.
  - SQLite persistence with SQLAlchemy models.
  - React news page with card‑based layout, images, and external links.

- **Animated landing page**
  - Driver/car carousel with team‑themed liveries.
  - Letter‑by‑letter title reveal and F1‑style theming.

- **End‑to‑end data & modeling pipeline**
  - Automated data acquisition for schedules, results, weather, and circuits.
  - Reproducible feature engineering for driver, team, and track performance.
  - Scripted model training and evaluation using XGBoost / LightGBM.

---

## Tech Stack

- **Frontend**
  - React (Create React App) app.
  - Client‑side routing with page components under `frontend/src/pages/`.
  - Utility‑first styling (Tailwind‑style classes) for F1‑themed UI.

- **Backend**
  - Python 3 / Flask REST API.
  - Flask Blueprints for predictor, news, and health endpoints.
  - APScheduler for scheduled data and news jobs.
  - SQLAlchemy ORM + SQLite for news storage.

- **ML & data**
  - XGBoost, LightGBM, scikit‑learn for ranking and regression.
  - Pandas / NumPy for data processing.
  - Pickled models and encoders stored under `backend/models/`.

- **Data sources**
  - Ergast F1 API via `jolpi.ca` proxy (schedules, results).
  - Official Formula 1 RSS feed (`https://www.formula1.com/en/latest/all.xml`).
  - Circuit specs, weather, and schedule metadata CSVs under `backend/data/`.

---

## Architecture

- **Frontend (`frontend/`)**
  - Key pages:
    - `landing.js` – animated landing experience
    - `news.js` – news hub backed by the Flask API
    - `predictor.js` – upcoming and historical race predictor dashboards
  - CRA dev server at `http://localhost:3000` and proxy to the backend at `http://127.0.0.1:5001`.

- **Backend (`backend/`)**
  - `app.py` – Flask app factory, CORS, APScheduler, news and health endpoints
  - `predictor_api.py` – ML prediction endpoints and feature preparation utilities
  - `news_job.py` – periodic RSS ingestion and persistence
  - `models.py` – SQLAlchemy models (e.g., `NewsItem`) and database setup
  - `data/` – CSVs, JSON metadata, and feature matrices
  - `models/` – serialized ML models and encoders

- **ML & data pipeline**
  - Acquisition scripts: `schedule_scraper.py`, `fetch_all_races.py`, `fetch_weather.py`, `add_weather.py`
  - Feature engineering: `feature_engineering.py`, `update_features.py`
  - Training: `train_grid_predictor.py`, `train_rank_model.py`
  - Prediction service: `predictor_api.py` loads models and exposes REST endpoints

---

## Repository Layout

```text
backend/
	app.py                 # Flask app, blueprints, health + news APIs
	predictor_api.py       # ML prediction endpoints and feature prep
	news_job.py            # Job to fetch and persist news headlines
	schedule_scraper.py    # Writes current‑year schedule CSV from Ergast
	fetch_all_races.py     # Pulls historical race results
	fetch_weather.py       # Fetches weather data for races
	add_weather.py         # Joins weather onto race history
	feature_engineering.py # Builds model features from raw data
	update_features.py     # Maintains up‑to‑date feature matrices
	train_grid_predictor.py# Trains grid position model
	train_rank_model.py    # Trains finish‑position ranking models
	models.py              # SQLAlchemy ORM models, Base + NewsItem
	data/                  # CSVs, JSON metadata, and feature matrices
	models/                # Serialized ML models and encoders

frontend/
	src/
		App.js               # Top‑level SPA with nav and routing state
		pages/landing.js     # Animated landing experience
		pages/news.js        # News hub, backed by /api/news
		pages/predictor.js   # Race predictor + historical comparison UI
	public/cars/           # Car images used by landing animation
```

---

## Getting Started

### Prerequisites

- Python 3.10+ (recommended)
- Node.js 18+ and npm

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/formula1nsights.git
cd formula1nsights
```

### 2. Backend setup (Flask + ML API)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

python app.py
```

By default, the Flask app runs on `http://127.0.0.1:5001` and registers the predictor blueprint plus news and health endpoints.

**Optional environment variables for news alerts (if you enable SMS/email logic):**

- `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM`, `TWILIO_TO`
- `NEWS_EMAIL_USER`, `NEWS_EMAIL_PASS`, `SMTP_SERVER`, `SMTP_PORT`

These are read in `news_job.py`. The SMS/email send helpers are currently commented out; you can re‑enable them when you have credentials.

### 3. Frontend setup (React + Tailwind)

```bash
cd frontend
npm install
npm start
```

Create React App will start a dev server at `http://localhost:3000` and forward API requests to the backend via the proxy defined in `frontend/package.json`:

```json
"proxy": "http://127.0.0.1:5001"
```

Ensure the backend is running on port `5001` for the News and Predictor pages to work correctly.

---

## Data & Modeling Workflow

The modeling side of Formula 1nsights is built around reproducible scripts in `backend/`:

1. **Data acquisition**

- `schedule_scraper.py` – writes the current season schedule to `data/schedule_<YEAR>.csv`.
- `fetch_all_races.py` – downloads race history into `data/race_history.csv`.
- `fetch_weather.py` – pulls weather data for each race.
- `add_weather.py` – merges weather into race history.

2. **Feature engineering**

- `feature_engineering.py` – creates driver/team/circuit‑level features (form, podium rates, consistency, etc.).
- `update_features.py` – keeps `predictor_features.csv` and `upcoming_features.csv` current for both historical and future races.

3. **Model training**

- `train_grid_predictor.py` – trains a grid position model using engineered features.
- `train_rank_model.py` – trains ranking models for race finish positions, including a pre‑qualifying version for future races.

4. **Serving**

- `predictor_api.py` loads the trained models and encoders from `backend/models/` and consumes the CSVs in `backend/data/`.
- Endpoints expose predictions for both past and upcoming races, which the React Predictor page consumes.

If you are only using the app as a consumer (not retraining models), ensure that the `data/` and `models/` folders contain the expected CSVs and `.pkl` files.

---

## Background Jobs & Operations

- `app.py` configures APScheduler jobs to keep race data and news fresh.
- When run as `__main__`, the app periodically triggers `news_job` to pull the latest headlines.

In production you would typically:

- Run the Flask app under a process manager (e.g., gunicorn + systemd).
- Run APScheduler jobs as part of the same process, or move them to a worker/cron environment.

---

## Health Checks & Debugging

- **Backend health**
  - `/api/health` and `/health` (in `app.py`)

- **News ingestion**
  - `/api/news` or `/news` to verify RSS ingestion and DB writes

- **Model configuration**
  - `/model_info` and `/validate_future_features` to confirm that the API sees the expected feature sets and future race features

If the Predictor page shows fallback races or no predictions, confirm that:

- The backend is running on `http://127.0.0.1:5001`.
- `available_races` and `predict_*` endpoints return non‑error JSON.
- Required CSVs (`predictor_features.csv`, `upcoming_features.csv`) and model files exist under `backend/data/` and `backend/models/`.

---

## Future Improvements

- Expand historical analysis (per‑race error metrics, driver consistency dashboards).
- Add user‑configurable notifications (Twilio/email) when new predictions are available.
- Package model training as a repeatable pipeline (e.g., CLI or notebook walkthroughs).
- Deploy backend and frontend as a cloud‑hosted app with a managed database.
- Add pre‑race feature engineering for tyre degradation profiles, one‑stop vs two‑stop strategy likelihood, and clean‑air race pace to better capture race‑day dynamics.

---

## License

See `LICENSE` for details.
