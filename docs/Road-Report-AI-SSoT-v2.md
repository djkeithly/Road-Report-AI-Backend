# AGENT-READY PRODUCT SPECIFICATION

## Road Report AI

Crash Risk Prediction Platform

- **Version:** 2.0 - Current Repositories Alignment
- **Last Updated:** March 2026
- **Source of Truth:** This document supersedes older SSoT drafts for implementation planning.

---

## Table of Contents

1. Product Overview
2. Tech Stack
3. Repository Structure and Status
4. Data Contracts
5. Backend API Reference
6. Frontend Integration Status
7. Error Handling Standards
8. Non-Functional Requirements
9. Environment Variables
10. Delivery Phases and Gaps
11. Agent Instructions and Constraints

---

## 1) Product Overview

Road Report AI predicts crash risk for road segments using historical crash data and contextual signals (weather and road metadata).

### 1.1 Target Users

- First responders
- City planners
- Commuters (later-phase productization)

### 1.2 Current Product Reality

- Backend API is running and trained-model aware.
- Frontend currently contains strong UI scaffolding and mock-driven reporting/chat behavior.
- Backend and frontend contract shape is partially aligned and now documented explicitly below.

---

## 2) Tech Stack

| Layer | Technology | Status |
|---|---|---|
| Frontend | Vue 3 + Vite + TypeScript | Implemented |
| Mapping | MapLibre + OSM/Overpass + Nominatim | Implemented in frontend |
| Backend | Python 3.13 + FastAPI | Implemented |
| Database | SQLAlchemy 2.0 async + SQLite/PostgreSQL URL support | Implemented |
| Validation | Pydantic v2 | Implemented |
| ML | PyTorch + Pandas + NumPy | Implemented baseline pipeline |
| External APIs | weather.gov | Implemented backend service |

---

## 3) Repository Structure and Status

### 3.1 Backend (`Road-Report-AI-Backend`)

```text
app/
  api/routes/
    health.py                  # /health, /health/model
    risk.py                    # /risk/predict, /risk/model-metrics
  config.py                    # Env loading + safe defaults for blank values
  database.py                  # Async engine/session
  models/prediction.py         # PredictionRecord scaffold
  schemas/
    risk.py                    # Request/response models
    health.py                  # Model metadata response model
  services/
    weather.py                 # weather.gov client + fallback
    risk.py                    # Runtime model inference + heuristic fallback
ml/
  preprocessing.py             # CSV loading/encoding
  model.py                     # BaselineCrashRiskModel
  training.py                  # Train + threshold tuning + metadata export
  evaluate.py                  # Report runner
  artifacts/                   # latest-model.pt + latest-model.meta.json (local)
docs/
  ssot-addendum-v1.md
  Road-Report-AI-SSoT-v2.md
```

### 3.2 Frontend (`Road-Report-AI-Frontend`)

```text
road-report-site/src/
  router/index.ts              # Home, Chat, Report, About, Account routes
  views/
    HomeView.vue               # Search + map exploration
    ReportView.vue             # Report layout (currently mock prediction flow)
    ChatView.vue               # Chat layout (currently mock assistant responses)
  components/
    MapDisplay.vue             # Overpass fetch and map render
  types/risk.ts                # Rich risk score contracts
```

### 3.3 Capability Matrix

| Capability | Backend | Frontend |
|---|---|---|
| Health check | Yes | N/A |
| Model metadata endpoint | Yes | Not yet consumed |
| Risk prediction endpoint | Yes | Not yet fully wired (mocks still used) |
| Weather enrichment | Yes | Display-layer references |
| Interactive road map | N/A | Yes |
| AI chat panel UI | N/A | Yes (mock data) |
| End-to-end live backend integration | Partial | Partial |

---

## 4) Data Contracts

### 4.1 Risk Predict Request

Required:
- `latitude` (float)
- `longitude` (float)

Optional:
- `road_name`
- `segment`
- `road_class`
- `weather_condition`
- `query_time_iso`

### 4.2 Risk Predict Response

Includes:
- `risk_score` (0-1)
- `total` (0-100)
- `tier`
- `components` (C/A/E/T breakdown)
- `summary`, `advice`
- `weather`
- `warnings`
- coordinates in both flat and object form

---

## 5) Backend API Reference

Base URL: `/api/v1`

- `GET /health`
- `GET /health/model`
- `GET /risk/model-metrics`
- `POST /risk/predict`

---

## 6) Frontend Integration Status

### Implemented in UI

- Route-level navigation and page scaffolds
- Interactive map fetching roads via Overpass API
- Report and chat interfaces with mock data models

### Not Yet Fully Wired

- `ReportView.vue` to `POST /api/v1/risk/predict`
- `ChatView.vue` to backend conversational/analysis endpoint
- Frontend model health/metrics display from `/risk/model-metrics`

---

## 7) Error Handling Standards

All backend API errors return:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "details": []
  }
}
```

Codes currently used:
- `VALIDATION_ERROR`
- `UNAUTHORIZED`
- `NOT_FOUND`
- `RATE_LIMITED`
- `INTERAL_ERROR` (kept for compatibility with prior SSoT spelling)

---

## 8) Non-Functional Requirements

- Async DB and async external API calls in route path
- Model and metadata health endpoints available for observability
- Config boot resilience when `.env` keys are blank (safe defaults)
- Local training/evaluation tooling for repeatability

---

## 9) Environment Variables

Backend:

- `DATABASE_URL`
- `API_V1_PREFIX`
- `DEBUG`
- `CORS_ORIGINS_CSV`
- `GOOGLE_MAPS_API_KEY`
- `WEATHER_API_BASE_URL`
- `WEATHER_USER_AGENT`
- `WEATHER_TIMEOUT_SECONDS`
- `MODEL_FILE_PATH`
- `MODEL_VERSION`

---

## 10) Delivery Phases and Gaps

### Completed/Available

- Boilerplate backend/frontend structures
- Trained-model pipeline and runtime inference wiring
- Model threshold tuning and exported metadata
- Health + model metrics API endpoints

### Next Recommended

1. Wire frontend report page to live backend risk endpoint.
2. Replace chat mock output with backend-driven responses.
3. Add auth/OAuth scope and protected routes if required by deployment plan.
4. Add migrations baseline and API test suite for new endpoints.

---

## 11) Agent Instructions and Constraints

- Follow `.cursorrules`.
- If spec ambiguity appears, use simpler implementation and document as `SPEC QUESTION`.
- If architecture must diverge, document as `ARCH DECISION`.
- Do not hardcode secrets; use environment variables only.

