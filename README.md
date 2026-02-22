# Road Report AI – Backend

Backend API for the Road Report AI crash risk prediction system. Built with FastAPI, PostgreSQL (or SQLite for local dev), and designed to integrate with a PyTorch ML model.

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
├── main.py          # FastAPI app, CORS, routers
├── config.py        # Settings from .env
├── database.py      # SQLAlchemy engine & session
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
| `DATABASE_URL` | PostgreSQL connection string | `sqlite:///./test.db` |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key (optional) | - |

## Next Steps

1. **Database models** – Add SQLAlchemy models in `app/models/` when you need to persist data (e.g. predictions, road segments).
2. **AI model** – Replace the placeholder in `app/services/risk.py` with your PyTorch model inference.
3. **Auth** – Add OAuth when ready (e.g. FastAPI OAuth2, Auth0).
4. **More inputs** – Extend `RiskRequest` in `app/schemas/risk.py` with fields like `road_type`, `time_of_day`, `weather_condition` as your model needs them.
