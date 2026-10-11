# Credit Card Fraud Detection with a Deep AutoEncoder (PyOD)

MSCS-633 — Assignment 4

An unsupervised deep-learning fraud detector built with the
[PyOD](https://pyod.readthedocs.io/) `AutoEncoder`. The model learns the
pattern of **genuine** credit-card transactions and flags transactions with a
high **reconstruction error** as likely fraud.

## Dataset

Anonymized credit-card transactions dataset referenced in the assignment:
https://www.kaggle.com/datasets/whenamancodes/fraud-detection

- 284,807 transactions, 492 of them fraudulent (0.173% — highly imbalanced)
- Features `V1`–`V28` are PCA-transformed (confidential), plus `Amount`
- `Class` = label (0 = genuine, 1 = fraud), used **only** for evaluation

Place `creditcard.csv` in this folder before running.

## How it works

1. Split the data into train/test (stratified, 70/30).
2. Train the AutoEncoder on **genuine transactions only** (semi-supervised
   novelty detection). The network never sees fraud.
3. Score the held-out test set: high reconstruction error = anomaly = fraud.
4. Evaluate with ROC-AUC, PR-AUC, a confusion matrix, and a classification
   report; save diagnostic plots to `outputs/`.

## Setup & run

```bash
python3 -m pip install -r requirements.txt
python3 fraud_detection_autoencoder.py
```
