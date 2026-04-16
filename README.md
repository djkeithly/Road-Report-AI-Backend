# Road Report AI - Backend

FastAPI backend for crash-risk prediction using one production model: the group logistic-regression model artifacts.

## 1) Setup and run (Windows)

```bash
py -3.13 -m venv venv313
.\venv313\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

App URLs:
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## 2) Single model source of truth

The backend uses only these artifacts:
- `app/services/logistic_regression_model.pth`
- `app/services/feature_columns.pkl`

They are loaded by `app/services/risk.py` at runtime.

When the model is retrained:
1. train locally
2. replace those two files
3. commit and push
4. Render auto-redeploys with the updated model

## 3) Environment file

Copy `.env.example` to `.env` and set values as needed.

Important keys:
- `DATABASE_URL`
- `API_V1_PREFIX`
- `DEBUG`
- `CORS_ORIGINS_CSV`
- `WEATHER_API_BASE_URL`
- `WEATHER_USER_AGENT`
- `WEATHER_TIMEOUT_SECONDS`
- `MODEL_FILE_PATH`
- `FEATURE_COLUMNS_PATH`
- `MODEL_VERSION`

Production CORS example:

```env
CORS_ORIGINS_CSV=https://road-report-ai-frontend.vercel.app,https://www.yourdomain.com
```

## 4) Endpoints

| Method | Endpoint |
|---|---|
| GET | `/` |
| GET | `/api/v1/health` |
| GET | `/api/v1/health/model` |
| GET | `/api/v1/risk/model-metrics` |
| POST | `/api/v1/risk/predict` |
| GET | `/api/v1/weather/forecast` |

## 5) Render deployment

- Branch: `main`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Required env variables:

```env
API_V1_PREFIX=/api/v1
DEBUG=false
CORS_ORIGINS_CSV=https://road-report-ai-frontend.vercel.app
MODEL_FILE_PATH=app/services/logistic_regression_model.pth
FEATURE_COLUMNS_PATH=app/services/feature_columns.pkl
MODEL_VERSION=log-reg-v1
WEATHER_API_BASE_URL=https://api.weather.gov
WEATHER_USER_AGENT=(roadreportai.prod, your-email@example.com)
WEATHER_TIMEOUT_SECONDS=6.0
```
