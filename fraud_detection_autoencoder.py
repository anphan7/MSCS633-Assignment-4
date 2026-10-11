from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")  # headless backend so plots save to file without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_FILE = "creditcard.csv"       # dataset in the project folder
OUTPUT_DIR = "outputs"             # where plots/artifacts are written
RANDOM_STATE = 42                  # reproducibility
TEST_SIZE = 0.30                   # fraction of data held out for evaluation


def load_dataset(path: str) -> pd.DataFrame:
    """Load the credit-card dataset and clean the label column.

    Some public mirrors export ``Class`` as a quoted string (e.g. "'0'").
    This normalizes it back to an integer so downstream code is simple.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Download the dataset from "
            "https://www.kaggle.com/datasets/whenamancodes/fraud-detection "
            "and place creditcard.csv in this folder."
        )

    df = pd.read_csv(path)
    # Normalize the label column to plain integers (handles "'0'"/"'1'" variants).
    df["Class"] = (
        df["Class"].astype(str).str.replace("'", "", regex=False).astype(int)
    )
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a short, human-readable overview of the data for the report."""
    n_rows, n_cols = df.shape
    n_fraud = int(df["Class"].sum())
    n_genuine = n_rows - n_fraud
    fraud_pct = 100.0 * n_fraud / n_rows

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Transactions (rows) : {n_rows:,}")
    print(f"Features (columns)  : {n_cols - 1}  (+ 1 label column 'Class')")
    print(f"Genuine (Class=0)   : {n_genuine:,}")
    print(f"Fraud   (Class=1)   : {n_fraud:,}")
    print(f"Fraud ratio         : {fraud_pct:.3f}%  (highly imbalanced)")
    print()


def build_model(contamination: float) -> AutoEncoder:
    """Create a PyOD AutoEncoder configured for this problem.

    ``contamination`` is the expected proportion of anomalies; PyOD uses it to
    place the decision threshold on the reconstruction-error scores.
    """
    return AutoEncoder(
        contamination=contamination,     # tolerated false-alarm rate -> sets threshold
        hidden_neuron_list=[20, 10],     # encoder: 29 -> 20 -> 10 (bottleneck), mirrored to decode
        hidden_activation_name="relu",
        epoch_num=30,                    # passes over the training data
        batch_size=256,                  # mini-batch size (fast on ~200k rows)
        lr=1e-3,                         # Adam learning rate
        dropout_rate=0.2,                # regularization to avoid over-fitting
        batch_norm=True,
        preprocessing=False,             # we scale the data ourselves (below)
        random_state=RANDOM_STATE,
        verbose=1,
    )


def plot_reconstruction_error(scores: np.ndarray, y_true: np.ndarray, path: str) -> None:
    """Histogram of anomaly (reconstruction-error) scores split by true class."""
    plt.figure(figsize=(8, 5))
    plt.hist(scores[y_true == 0], bins=60, alpha=0.7, label="Genuine", color="#2a9d8f")
    plt.hist(scores[y_true == 1], bins=60, alpha=0.9, label="Fraud", color="#e63946")
    plt.yscale("log")  # log scale: fraud is rare, otherwise invisible
    plt.xlabel("Anomaly score (reconstruction error)")
    plt.ylabel("Number of transactions (log scale)")
    plt.title("AutoEncoder Reconstruction Error: Genuine vs. Fraud")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, auc: float, path: str) -> None:
    """Receiver Operating Characteristic curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#1d3557", lw=2, label=f"AutoEncoder (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    """Confusion-matrix heatmap for the chosen decision threshold."""
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (test set)")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ["Genuine", "Fraud"])
    plt.yticks(ticks, ["Genuine", "Fraud"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # Annotate each cell with its count.
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12, fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and summarize the data -------------------------------------- #
    df = load_dataset(DATA_FILE)
    describe_dataset(df)

    X = df.drop(columns=["Class"]).values  # features: V1..V28, Amount
    y = df["Class"].values                 # labels (used only for evaluation)

    # 2) Train / test split (stratified to keep the same fraud ratio) ----- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training transactions : {len(X_train):,}")
    print(f"Test transactions     : {len(X_test):,}")
    print()

    # 3) Keep only GENUINE transactions for training --------------------- #
    # Semi-supervised anomaly detection: the AutoEncoder learns what "normal"
    # looks like from genuine transactions only. It never sees fraud, so frauds
    # reconstruct poorly at test time and stand out as high-error anomalies.
    X_train_genuine = X_train[y_train == 0]
    print(f"Genuine-only training set: {len(X_train_genuine):,} transactions")
    print()

    # 4) Feature scaling -------------------------------------------------- #
    # Standardize to zero-mean / unit-variance. The scaler is fit ONLY on the
    # genuine training data to avoid leaking information from the test set.
    scaler = StandardScaler()
    X_train_genuine = scaler.fit_transform(X_train_genuine)
    X_test = scaler.transform(X_test)

    # 5) Build and train the AutoEncoder --------------------------------- #
    # ``contamination`` here is the tolerated false-alarm rate: PyOD places the
    # decision threshold at that upper percentile of the genuine training errors.
    contamination = 0.01
    print("=" * 70)
    print("TRAINING AUTOENCODER (on genuine transactions only)")
    print("=" * 70)
    print(f"Decision threshold percentile (contamination): {contamination:.3f}")
    print()

    model = build_model(contamination)
    start = time.time()
    model.fit(X_train_genuine)  # labels are NOT used during training
    print(f"\nTraining completed in {time.time() - start:.1f} seconds.\n")

    # 5) Score the held-out test set -------------------------------------- #
    # Higher score == larger reconstruction error == more likely fraud.
    test_scores = model.decision_function(X_test)
    test_pred = model.predict(X_test)  # 0/1 using PyOD's contamination threshold

    # 6) Evaluate --------------------------------------------------------- #
    roc_auc = roc_auc_score(y_test, test_scores)
    avg_precision = average_precision_score(y_test, test_scores)
    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    print(f"ROC-AUC score            : {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")
    print()
    print("Confusion matrix:")
    print(f"  True Negatives  (genuine kept) : {tn:,}")
    print(f"  False Positives (false alarms) : {fp:,}")
    print(f"  False Negatives (missed fraud) : {fn:,}")
    print(f"  True Positives  (caught fraud) : {tp:,}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test, test_pred, target_names=["Genuine", "Fraud"], digits=4
        )
    )

    # 7) Save plots for the report ---------------------------------------- #
    plot_reconstruction_error(
        test_scores, y_test, os.path.join(OUTPUT_DIR, "reconstruction_error.png")
    )
    plot_roc_curve(y_test, test_scores, roc_auc, os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plot_confusion_matrix(cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Precision-recall curve (useful for imbalanced data).
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="#6a4c93", lw=2,
             label=f"AutoEncoder (PR-AUC = {avg_precision:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall.png"), dpi=130)
    plt.close()

    print(f"Plots saved to '{OUTPUT_DIR}/' (reconstruction_error.png, roc_curve.png, "
          "confusion_matrix.png, precision_recall.png)")
    print("\nDone.")


if __name__ == "__main__":
    main()
