"""Evaluation report script for trained crash-risk models."""

import argparse

from ml.training import run_training_pipeline


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for evaluation command."""
    parser = argparse.ArgumentParser(description="Evaluate crash risk model with report.")
    parser.add_argument("--csv-path", default="csv/TrainingData.csv")
    parser.add_argument("--output-path", default="ml/artifacts/latest-model.pt")
    parser.add_argument("--row-limit", type=int, default=250000)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main() -> None:
    """Run training + evaluation and print concise quality report."""
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

    print("Evaluation report:")
    print(f"rows_used={metrics['rows_used']}")
    print(f"input_size={metrics['input_size']}")
    print(f"threshold={metrics['threshold']:.2f}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
