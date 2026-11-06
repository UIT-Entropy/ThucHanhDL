import argparse

from training.experiments import run_assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-key", choices=["topic", "sentiment"], default="topic")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 0:
        parser.error("epochs, batch-size and learning-rate must be positive; patience must be nonnegative")
    run_assignment(args.assignment, args.data_dir, args.output_dir, args.epochs, args.batch_size, args.learning_rate, args.patience, args.seed, args.label_key, args.device)


if __name__ == "__main__":
    main()
