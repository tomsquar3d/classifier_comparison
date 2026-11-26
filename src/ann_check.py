"""Diagnostic for the neural net: is it stopping too early / learning too slowly?

Timing test (2026-11-26) showed the MLP only reaching ~0.70 training
accuracy on HIGGS, with the smallest net winning — signs of undertraining.
This trains one mid-sized net (2 x 128) on the HIGGS 80/20 trial-1 training
split under a few learning-rate / early-stopping settings and reports how
many epochs each ran, its training accuracy and its held-out accuracy.
Not part of the results; just to choose the ANN settings.

Usage (DSMLP container, cogs118a env):
    python src/ann_check.py                 # HIGGS
    python src/ann_check.py --dataset adult
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from models import MLP  # noqa: E402

# (label, optimizer, lr, momentum, patience, max_epochs)
SETTINGS = [
    ("current: SGD lr 0.01, patience 10", torch.optim.SGD, 0.01, 0.9, 10, 200),
    ("SGD lr 0.01, patience 30",          torch.optim.SGD, 0.01, 0.9, 30, 400),
    ("SGD lr 0.05, patience 10",          torch.optim.SGD, 0.05, 0.9, 10, 200),
    ("SGD lr 0.1,  patience 10",          torch.optim.SGD, 0.1,  0.9, 10, 200),
    ("SGD lr 0.1,  patience 30",          torch.optim.SGD, 0.1,  0.9, 30, 400),
    ("Adam lr 0.001, patience 10",        torch.optim.Adam, 0.001, None, 10, 200),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="higgs", choices=C.DATASETS)
    ap.add_argument("--hidden", default="128,128")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))

    data = np.load(C.PROCESSED_DIR / f"{args.dataset}.npz")
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    tr = data["train_idx__80_20__t0"]
    # Inner split of the training data only (the test split is never touched)
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X[tr], y[tr], test_size=0.2, stratify=y[tr], random_state=0)
    sc = StandardScaler().fit(X_fit)
    X_fit = sc.transform(X_fit).astype(np.float32)
    X_hold = sc.transform(X_hold).astype(np.float32)
    print(f"{args.dataset}: fit on {len(y_fit):,} rows (10% of those used for "
          f"early stopping), held out {len(y_hold):,}; hidden={hidden}\n")
    print(f"{'setting':38s} {'epochs':>6s} {'best':>5s} {'train':>6s} "
          f"{'held-out':>8s} {'time':>6s}")

    for label, opt, lr, mom, patience, max_ep in SETTINGS:
        torch.manual_seed(0)
        kw = {"optimizer__momentum": mom} if mom is not None else {}
        net = NeuralNetClassifier(
            MLP, module__d_in=X.shape[1], module__hidden=hidden,
            criterion=torch.nn.CrossEntropyLoss, optimizer=opt, lr=lr,
            max_epochs=max_ep, batch_size=C.ANN_BATCH_SIZE,
            train_split=ValidSplit(C.ANN_VALID_FRAC, stratified=True,
                                   random_state=0),
            callbacks=[EarlyStopping(monitor="valid_loss", patience=patience,
                                     load_best=True)],
            iterator_train__shuffle=True,
            iterator_valid__batch_size=C.ANN_PREDICT_BATCH_SIZE,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0, **kw)
        t0 = time.perf_counter()
        net.fit(X_fit, y_fit)
        took = time.perf_counter() - t0
        losses = net.history[:, "valid_loss"]
        best_ep = int(np.argmin(losses)) + 1
        tr_acc = float((net.predict(X_fit) == y_fit).mean())
        ho_acc = float((net.predict(X_hold) == y_hold).mean())
        print(f"{label:38s} {len(net.history):6d} {best_ep:5d} {tr_acc:6.3f} "
              f"{ho_acc:8.3f} {took:5.0f}s", flush=True)

    print("\nepochs = epochs run before early stopping; best = epoch with the "
          "lowest validation loss (the weights kept).")


if __name__ == "__main__":
    main()
