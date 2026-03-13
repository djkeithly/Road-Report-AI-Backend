# Road Report AI SSoT Addendum v1

This addendum records implementation clarifications and decisions made while aligning the backend to the SSoT.

The consolidated implementation-aligned spec now lives in `docs/Road-Report-AI-SSoT-v2.md`.

## Change Scope

- Backend repository: `Road-Report-AI-Backend-main`
- Frontend reference: [djkeithly/Road-Report-AI-Frontend](https://github.com/djkeithly/Road-Report-AI-Frontend)
- Explicitly ignored frontend `Weekly Progress` content as requested.

## API Contract Clarifications

### `POST /api/v1/risk/predict`

- Request now supports:
  - `latitude`, `longitude` (required)
  - `road_name`, `segment`, `weather_condition`, `road_class`, `query_time_iso` (optional)
- Response now includes:
  - `risk_score` in 0-1 range (backward compatible)
  - `total` in 0-100 range (frontend display ready)
  - `tier`, `components`, `summary`, `advice`, `weather`, `warnings`
  - both `latitude`/`longitude` and `coordinates.lat`/`coordinates.lng`

## Error Envelope Standard

All API errors are normalized to:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "details": []
  }
}
```

Codes currently emitted:

- `VALIDATION_ERROR`
- `UNAUTHORIZED`
- `NOT_FOUND`
- `RATE_LIMITED`
- `INTERAL_ERROR`

## Architecture Decisions

- ARCH DECISION: Keep deterministic scoring scaffold in API service until trained model artifacts are available, so frontend integration can proceed without blocking on ML training.
- ARCH DECISION: Use weather.gov hourly forecast (`/points` -> `forecastHourly`) with graceful fallback payloads to satisfy non-blocking and reliability requirements.
- ARCH DECISION: Preserve both normalized (`risk_score`) and display (`total`) scoring outputs to bridge backend and frontend representation differences.
- ARCH DECISION: Add minimal SQLAlchemy `PredictionRecord` scaffold for persistence readiness without forcing immediate migration rollout.

## Spec Questions

- SPEC QUESTION: SSoT describes weather and road integrations but does not define strict backend endpoint set beyond `/health` and `/risk/predict`; current implementation enriches `risk/predict` first and leaves future map proxy endpoints for follow-up.
- SPEC QUESTION: OAuth is listed in stack and constraints, but no auth flow, provider, scopes, or protected endpoints are defined yet.

## Spec Errors/Conflicts Noted

- SPEC ERROR: Error code list includes `INTERAL_ERROR` typo. Current backend keeps that code to remain spec-consistent and avoid undocumented divergence.
- SPEC ERROR: SSoT section numbering repeats/inconsistently labels the agent instructions section.

## Follow-up Recommendations

1. Define a strict API contract for map and weather endpoints (if frontend should call backend only).
2. Add Alembic migration baseline for `predictions` table.
3. Replace deterministic scaffold with PyTorch inference loaded from `MODEL_FILE_PATH`.

## Training Pipeline Update

- Training data wired from `csv/TrainingData.csv` into `ml.training` CLI.
- ARCH DECISION: Use weighted binary logits loss (`BCEWithLogitsLoss` with `pos_weight`) due to heavy crash/non-crash imbalance.
- Evaluation now prints `accuracy`, `precision`, `recall`, and `f1` to avoid misleading accuracy-only reporting.
- ARCH DECISION: Tune holdout threshold for best F1 and persist metadata to `latest-model.meta.json` for runtime inference consistency.
- ARCH DECISION: Expose model artifact status and latest metadata metrics via `GET /api/v1/health/model` for operational visibility.
- ARCH DECISION: Expose frontend-focused model metrics via `GET /api/v1/risk/model-metrics` to decouple UI cards from operational health endpoints.
