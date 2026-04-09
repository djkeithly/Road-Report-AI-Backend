# Road Report AI – Backend

Backend API for the Road Report AI crash risk prediction system. Built with FastAPI, PostgreSQL (async via asyncpg) or SQLite (async via aiosqlite), SQLAlchemy 2.0, and designed to integrate with a PyTorch ML model.

## Quick Start

```bash
# Create and activate venv (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the API
python main.py
# or: uvicorn app.main:app --reload
```

API runs at **http://localhost:8000**  
Swagger docs: **http://localhost:8000/docs**

## Project Structure

```
app/
├── main.py          # FastAPI app, lifespan, CORS, routers
├── config.py        # Settings from .env (supports async DB URLs)
├── database.py      # Async SQLAlchemy engine & session (asyncpg/aiosqlite)
├── api/
│   ├── deps.py      # Shared dependencies (get_db)
│   └── routes/
│       ├── health.py
│       └── risk.py  # Risk prediction endpoint
├── schemas/         # Pydantic request/response models
│   └── risk.py
└── services/        # Business logic
    └── risk.py      # TODO: wire to PyTorch model
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/risk/predict` | Get crash risk score for coordinates |
| GET | `/api/v1/weather/forecast` | Get the current weather |

### Example: Risk Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/risk/predict" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 30.2672, "longitude": -97.7431}'
```

## Environment Variables

Create a `.env` file (see `.env.example` if provided):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | DB connection string (`postgresql://` or `sqlite://`) | `sqlite:///./test.db` |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key (optional) | - |

For PostgreSQL, use `postgresql://user:pass@host:5432/dbname`; it is converted to `postgresql+asyncpg://` for async. For SQLite, `sqlite:///./test.db` is converted to `sqlite+aiosqlite://`.

## Next Steps

1. **Database models** – Add SQLAlchemy models in `app/models/` when you need to persist data (e.g. predictions, road segments).
2. **AI model** – Replace the placeholder in `app/services/risk.py` with your PyTorch model inference.
3. **Auth** – Add OAuth when ready (e.g. FastAPI OAuth2, Auth0).
4. **More inputs** – Extend `RiskRequest` in `app/schemas/risk.py` with fields like `road_type`, `time_of_day`, `weather_condition` as your model needs them.
