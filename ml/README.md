# ML Module

This folder provides a baseline structure for model development.

- `preprocessing.py`: feature preprocessing and training dataset shaping.
- `model.py`: PyTorch model definitions and model loading helpers.
- `training.py`: model training loop entry points.
- `artifacts/`: persisted model checkpoints (`.pt` files).

## Notes

- Keep training code decoupled from API-serving code.
- Export model artifacts to `ml/artifacts/` and reference via `MODEL_FILE_PATH`.
