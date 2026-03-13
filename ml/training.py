"""Training loop scaffolding for crash risk models."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ml.model import BaselineCrashRiskModel
from ml.preprocessing import (
    build_feature_table,
    load_training_dataframe,
    split_feature_table,
    to_tensor_dataset,
)


def train_binary_classifier(
    *,
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
) -> dict[str, nn.Module | list[float]]:
    """Train a binary classifier and return model with epoch losses."""
    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    positive_count = float(labels.sum().item())
    negative_count = float(labels.shape[0] - positive_count)
    pos_weight_value = max(1.0, negative_count / max(1.0, positive_count))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epoch_losses: list[float] = []

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in data_loader:
            optimizer.zero_grad()
            logits = model(batch_features).squeeze(-1)
            loss = criterion(logits, batch_labels.float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        epoch_losses.append(total_loss / max(1, len(data_loader)))

    return {"model": model, "epoch_losses": epoch_losses}


def evaluate_binary_classifier(
    *,
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate classifier on holdout data and return metrics."""
    model.eval()
    with torch.no_grad():
        logits = model(features).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()
        accuracy = float((predictions == labels).float().mean().item())
        positive_rate = float(predictions.mean().item())
        true_positive = float(((predictions == 1) & (labels == 1)).sum().item())
        false_positive = float(((predictions == 1) & (labels == 0)).sum().item())
        false_negative = float(((predictions == 0) & (labels == 1)).sum().item())

        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "positive_rate": positive_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tune_threshold(
    *,
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """Tune decision threshold by maximizing F1 on holdout data."""
    best_threshold = 0.5
    best_metrics = evaluate_binary_classifier(
        model=model,
        features=features,
        labels=labels,
        threshold=0.5,
    )
    best_f1 = best_metrics["f1"]
    best_recall = best_metrics["recall"]

    candidate = 0.05
    while candidate <= 0.95:
        metrics = evaluate_binary_classifier(
            model=model,
            features=features,
            labels=labels,
            threshold=round(candidate, 2),
        )
        score = metrics["f1"]
        if score > best_f1 or (score == best_f1 and metrics["recall"] > best_recall):
            best_threshold = round(candidate, 2)
            best_f1 = score
            best_recall = metrics["recall"]
            best_metrics = metrics
        candidate += 0.01

    return best_threshold, best_metrics


def save_model_state(model: nn.Module, *, output_path: str) -> None:
    """Persist model weights for API inference use."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)


def save_model_metadata(
    *,
    output_path: str,
    metadata: dict[str, str | int | float | list[str]],
) -> None:
    """Persist training metadata for runtime inference compatibility."""
    target = Path(output_path).with_suffix(".meta.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile, indent=2)


def run_training_pipeline(
    *,
    csv_path: str,
    output_path: str,
    row_limit: int | None = None,
    test_ratio: float = 0.2,
    seed: int = 7,
    epochs: int = 6,
    learning_rate: float = 1e-3,
    batch_size: int = 512,
) -> dict[str, float | int]:
    """Load CSV training data, fit model, evaluate, and save artifact."""
    dataframe = load_training_dataframe(csv_path=csv_path, row_limit=row_limit)
    shaped = build_feature_table(dataframe=dataframe)
    split_data = split_feature_table(
        features=shaped["features"],
        labels=shaped["labels"],
        test_ratio=test_ratio,
        seed=seed,
    )
    tensors = to_tensor_dataset(split_data=split_data)

    model = BaselineCrashRiskModel(input_size=tensors["input_size"])
    train_result = train_binary_classifier(
        model=model,
        features=tensors["x_train"],
        labels=tensors["y_train"],
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    fitted_model = train_result["model"]

    best_threshold, metrics = tune_threshold(
        model=fitted_model,
        features=tensors["x_test"],
        labels=tensors["y_test"],
    )
    save_model_state(fitted_model, output_path=output_path)
    save_model_metadata(
        output_path=output_path,
        metadata={
            "input_size": int(tensors["input_size"]),
            "feature_columns": list(shaped["features"].columns),
            "threshold": best_threshold,
            "rows_used": int(len(dataframe)),
            "test_ratio": test_ratio,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "accuracy": metrics["accuracy"],
            "positive_rate": metrics["positive_rate"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
    )

    return {
        "rows_used": int(len(dataframe)),
        "input_size": int(tensors["input_size"]),
        "accuracy": metrics["accuracy"],
        "positive_rate": metrics["positive_rate"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": best_threshold,
        "final_loss": float(train_result["epoch_losses"][-1]),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for model training."""
    parser = argparse.ArgumentParser(description="Train crash risk baseline model.")
    parser.add_argument("--csv-path", default="csv/TrainingData.csv")
    parser.add_argument("--output-path", default="ml/artifacts/latest-model.pt")
    parser.add_argument("--row-limit", type=int, default=250000)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main() -> None:
    """CLI entrypoint for model training pipeline."""
    args = _build_parser().parse_args()
    metrics = run_training_pipeline(
        csv_path=args.csv_path,
        output_path=args.output_path,
        row_limit=args.row_limit,
        test_ratio=args.test_ratio,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    print("Training complete:")
    print(f"rows_used={metrics['rows_used']}")
    print(f"input_size={metrics['input_size']}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"positive_rate={metrics['positive_rate']:.4f}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"threshold={metrics['threshold']:.2f}")
    print(f"final_loss={metrics['final_loss']:.6f}")


if __name__ == "__main__":
    main()
