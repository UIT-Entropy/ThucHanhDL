import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, title):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        ax.plot(epochs, [item[key] for item in history], marker="o")
        ax.set(xlabel="Epoch", ylabel=label, title=f"{title} - Train {label}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scores(scores, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(["Accuracy", "Precision-macro", "Recall-macro", "F1-macro"], list(scores.values()))
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylim=(0, 1.1), title=f"{title} - Test metrics")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, title, class_names=None):
    class_names = list(range(10)) if class_names is None else class_names
    count = len(class_names)
    matrix = np.zeros((count, count), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    fig, ax = plt.subplots(figsize=(9, 7) if count <= 10 else (14, 12))
    chart = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(count), yticks=range(count), xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="True", title=f"{title} - Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=90)
    for row in range(count):
        for col in range(count):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.colorbar(chart, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_distribution(train_targets, test_targets, class_names):
    positions = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(positions - 0.2, np.bincount(train_targets, minlength=len(class_names)), width=0.4, label="Train")
    ax.bar(positions + 0.2, np.bincount(test_targets, minlength=len(class_names)), width=0.4, label="Test")
    ax.set(xticks=positions, xticklabels=class_names, ylabel="Images", title="VinaFood21 - Class distribution")
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()
