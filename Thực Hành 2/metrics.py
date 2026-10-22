import numpy as np


def evaluate(y_true, y_pred, num_classes=10, class_names=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    precision = []
    recall = []
    f1 = []
    print(f"Accuracy: {accuracy:.4f}")
    print("Class                 Precision  Recall  F1-score")
    for digit in range(num_classes):
        tp = np.sum((y_true == digit) & (y_pred == digit))
        fp = np.sum((y_true != digit) & (y_pred == digit))
        fn = np.sum((y_true == digit) & (y_pred != digit))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        name = class_names[digit] if class_names is not None else str(digit)
        print(f"{name:20s}  {p:.4f}     {r:.4f}  {f:.4f}")
    scores = {"accuracy": accuracy, "precision_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)), "f1_macro": float(np.mean(f1))}
    print(f"Precision-macro: {scores['precision_macro']:.4f}")
    print(f"Recall-macro: {scores['recall_macro']:.4f}")
    print(f"F1-macro: {scores['f1_macro']:.4f}")
    return scores
