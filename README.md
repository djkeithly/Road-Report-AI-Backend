# Road Report AI - Backend

Backend API for crash risk prediction with FastAPI, SQLAlchemy async, and a PyTorch model pipeline.

## 1) Setup and Run (Windows)

```bash
# Recommended Python version: 3.13
py -3.13 -m venv venv313
.\venv313\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

App URLs:
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## 2) Environment File

Copy `.env.example` to `.env` and set values as needed.

Important keys:
- `DATABASE_URL`
- `API_V1_PREFIX`
- `DEBUG`
- `CORS_ORIGINS_CSV`
- `WEATHER_API_BASE_URL`
- `WEATHER_USER_AGENT`
- `WEATHER_TIMEOUT_SECONDS`
- `MODEL_FILE_PATH="ml/artifacts/latest-model.pt"`
- `MODEL_VERSION="baseline-v1"`

Production CORS example:

```env
CORS_ORIGINS_CSV=https://road-report-ai-frontend.vercel.app,https://www.yourdomain.com
```

## 3) Train the Model

If `csv/TrainingData.csv` exists, train and export model artifacts:

```bash
.\venv313\Scripts\python -m ml.training --csv-path csv/TrainingData.csv --output-path ml/artifacts/latest-model.pt --row-limit 250000 --epochs 8 --batch-size 512
```

Optional evaluation report:

```bash
.\venv313\Scripts\python -m ml.evaluate --csv-path csv/TrainingData.csv --output-path ml/artifacts/latest-model.pt --row-limit 250000 --epochs 8 --batch-size 512
```

## 4) Test with Swagger

1. Start backend with `python main.py`
2. Open `http://localhost:8000/docs`
3. Test these endpoints:
   - `GET /api/v1/health`
   - `GET /api/v1/health/model`
   - `GET /api/v1/risk/model-metrics`
   - `POST /api/v1/risk/predict`

### Swagger request example (likely lower risk)

Use in `POST /api/v1/risk/predict`:

```json
{
  "latitude": 32.7767,
  "longitude": -96.7970,
  "road_name": "Main St",
  "segment": "Dallas city center",
  "road_class": "CITY STREET",
  "weather_condition": "1 - CLEAR"
}
```

### Swagger request example (likely higher risk)

```json
{
  "latitude": 32.7767,
  "longitude": -96.7970,
  "road_name": "I-35E",
  "segment": "Downtown Dallas",
  "road_class": "INTERSTATE",
  "weather_condition": "3 - RAIN"
}
```

## 5) API Endpoints

| Method | Endpoint |
|---|---|
| GET | `/` |
| GET | `/api/v1/health` |
| GET | `/api/v1/health/model` |
| GET | `/api/v1/risk/model-metrics` |
| POST | `/api/v1/risk/predict` |
