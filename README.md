## Formula 1nsights

An end‑to‑end Formula 1 analytics and news platform that combines a modern React front‑end with a Flask‑based ML inference API. Formula 1nsights automatically ingests race schedules and results, enriches them with circuit and weather data, trains gradient‑boosted ranking models, and serves:

- AI‑powered race predictions for upcoming and historical races
- Live F1 news pulled from the official Formula 1 RSS feed
- Dashboards that compare model predictions to actual race results

This project demonstrates:

- A full‑stack **Formula 1 analytics website** with React + Flask
- A **news aggregation system** using RSS parsing, APScheduler jobs, and SQLite for real‑time headlines
- **XGBoost‑based race result models** trained on Ergast historical data with features for driver performance, team competitiveness, and racetrack characteristics
- **Temporal weighting**, Monte Carlo simulations for uncertainty estimation, and temporal validation to reduce leakage
- **Categorical encoding** for 20+ drivers across 20+ circuits and F1‑themed Tailwind dashboards wired to REST endpoints

---

## High‑Level Architecture

- **Frontend** (folder: `frontend/`)
  - React (Create React App) single‑page app
  - Tailwind CSS for theming and F1‑style visuals
  - Pages:
    - `Landing` – animated car carousel and Formula 1nsights logo
    - `News` – F1 news hub backed by the Flask API
    - `Predictor` – tables for upcoming race predictions and historical analysis
  - Uses CRA dev proxy to talk to the backend at `http://127.0.0.1:5001`.

- **Backend** (folder: `backend/`)
  - Flask API with CORS enabled
  - APScheduler for scheduled background jobs
  - SQLAlchemy + SQLite (`news.db`) for persisting fetched news
  - ML inference endpoints powered by scikit‑learn, XGBoost, and LightGBM
  - Data & model assets stored under `backend/data/` and `backend/models/`

- **Data & Models**
  - Sourced from:
    - Ergast F1 API via a `jolpi.ca` proxy (race schedules & history)
    - Official Formula 1 RSS feed (`https://www.formula1.com/en/latest/all.xml`)
    - Weather and circuit metadata CSVs
  - Feature engineering and training scripts generate:
    - Grid position model (`xgb_grid_predictor.pkl`)
    - Finish position ranking models (`xgb_finish_ranker.pkl`, `xgb_finish_ranker_pre_qual.pkl`)
    - Label encoders and model metadata (`label_encoders_*.pkl`, `model_info.json`, `feature_info.json`)

---

## Key Features

### 1. Animated Landing Page

- Cycles through 20 drivers and car liveries with team‑specific lighting.
- Reveals the Formula 1nsights logo letter‑by‑letter.
- Tagline: "Latest F1 News and Race Predictor".

### 2. F1 News Hub

- Background job (`news_job.py`) parses the official Formula 1 RSS feed using `feedparser`.
- New articles are stored in SQLite via SQLAlchemy models defined in `models.py`.
- Flask routes (in `app.py`) expose:
  - `/api/news` – JSON of latest headlines, summaries, URLs, and images
  - `/news` – same payload without the `/api` prefix (useful for local testing)
- Frontend `News` page renders a responsive grid of cards with images, summaries, and "Read more" links.
- Optional hooks (commented in code) for Twilio SMS and email alerts via environment‑configured credentials.

### 3. Race Predictor & Historical Analysis

- Core prediction logic lives in `backend/predictor_api.py` and is registered as a Flask Blueprint.
- Models and encoders are loaded once at startup (`load_models()`), with support for both legacy and structured `data/` layouts.
- Enhanced feature preparation handles categorical encoders, temporal feature sets, smart defaults, and strict feature ordering.
- Monte Carlo–style resampling (via `get_prediction_confidence`) provides uncertainty estimates and confidence intervals for finish positions.

**Endpoints (Blueprint `predictor_api`)**

- `/available_races`
  - Returns a map of `season -> [races]` and a list of available seasons.
  - Used by the Predictor UI to populate season and Grand Prix dropdowns.

- `/predict_race_with_predicted_grid`
  - Two‑stage pipeline: first predicts grid positions, then predicts race finish given that grid.
  - Used for upcoming 2025 races in the Predictor "Next race" table.

- `/predict_finish_pos_round`
  - Post‑qualifying model for completed races (uses `predictor_features.csv`).
  - Can optionally return confidence intervals via bootstrap sampling.

- `/predict_historical_race_two_stage`
  - Re‑runs the two‑stage (grid → finish) pipeline on historical races.
  - Enables comparison between post‑qualifying and two‑stage predictions.

- `/compare_grid_predictions`
  - Compares grid predictions vs actual qualifying results, per driver.

- `/predict_finish_pos_future`
  - Pre‑qualifying or post‑qualifying ranking for upcoming races (driven by `upcoming_features.csv`).

- `/model_info`, `/validate_future_features`, `/health`
  - Utilities for debugging feature sets, model metadata, and service health.

The `Predictor` page combines these endpoints to show:

- Upcoming race predicted grid and finish order for each driver.
- Historical race breakdown with columns for:
  - Predicted vs actual grid
  - Post‑qualifying vs two‑stage predicted finish
  - Where the model over‑ or under‑performed.

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

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/formula1nsights.git
cd formula1nsights
```

### 2. Backend Setup (Flask + ML API)

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

### 3. Frontend Setup (React + Tailwind)

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

1. **Data Acquisition**
   - `schedule_scraper.py` – writes the current season schedule to `data/schedule_<YEAR>.csv`.
   - `fetch_all_races.py` – downloads race history into `data/race_history.csv`.
   - `fetch_weather.py` – pulls weather data for each race.
   - `add_weather.py` – merges weather into race history.

2. **Feature Engineering**
   - `feature_engineering.py` – creates driver/team/circuit‑level features (form, podium rates, consistency, etc.).
   - `update_features.py` – keeps `predictor_features.csv` and `upcoming_features.csv` current for both historical and future races.

3. **Model Training**
   - `train_grid_predictor.py` – trains a grid position model using engineered features.
   - `train_rank_model.py` – trains ranking models for race finish positions, including a pre‑qualifying version for future races.

4. **Serving**
   - `predictor_api.py` loads the trained models and encoders from `backend/models/` and consumes the CSVs in `backend/data/`.
   - Endpoints expose predictions for both past and upcoming races, which the React Predictor page consumes.

If you are only using the app as a consumer (not retraining models), ensure that the `data/` and `models/` folders contain the expected CSVs and `.pkl` files.

---

## Running Background Jobs

- `app.py` configures an APScheduler job to periodically call `fetch_all_races` and keep race data fresh.
- When run as `__main__`, it also starts a scheduler that periodically runs `news_job` to pull the latest headlines.

In production, you would typically:

- Run the Flask app under a process manager (e.g., gunicorn + systemd)
- Run APScheduler jobs as part of the same process, or externalize them to a worker/cron environment if needed.

---

## Health Checks & Debugging

- Backend:
  - `/api/health` and `/health` (in `app.py`)
  - `/health` (in `predictor_api.py`)
- News: `/api/news` or `/news` to verify RSS ingestion and DB writes.
- Models: `/model_info` and `/validate_future_features` to confirm that the API sees the expected feature sets and future race features.

If the Predictor page shows fallback races or no predictions, confirm that:

- The backend is running on `http://127.0.0.1:5001`.
- `available_races`, `predict_*` endpoints return non‑error JSON.
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
