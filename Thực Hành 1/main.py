import argparse

import numpy as np

from Load_data import MNISTDataset
from modeling import MLP1Layer, MLP3Layer


def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    precision = []
    recall = []
    f1 = []
    print(f"Accuracy: {accuracy:.4f}")
    print("Digit  Precision  Recall  F1-score")
    for digit in range(10):
        tp = np.sum((y_true == digit) & (y_pred == digit))
        fp = np.sum((y_true != digit) & (y_pred == digit))
        fn = np.sum((y_true == digit) & (y_pred != digit))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        print(f"{digit:5d}  {p:.4f}     {r:.4f}  {f:.4f}")
    print(f"Precision-macro: {np.mean(precision):.4f}")
    print(f"Recall-macro: {np.mean(recall):.4f}")
    print(f"F1-macro: {np.mean(f1):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá MLP trên MNIST")
    parser.add_argument("--model", choices=("1", "3", "both"), default="both")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = MNISTDataset(args.data_dir).load()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    models = []
    if args.model in ("1", "both"):
        models.append(("Bài 1: MLP 1 lớp", MLP1Layer(seed=args.seed)))
    if args.model in ("3", "both"):
        models.append(("Bài 2: MLP 3 lớp", MLP3Layer(seed=args.seed)))
    for title, model in models:
        print(f"\n{title}")
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
        evaluate(y_test, model.predict(x_test))


if __name__ == "__main__":
    main()
