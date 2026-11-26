"""Model factories: one place that turns a model name into an estimator.

build(name, n_features, device, seed) returns (estimator, param_grid).
The estimator is ready for sklearn's GridSearchCV; param_grid comes from
config.GRIDS (filtered where a value doesn't fit the dataset).

device="gpu" is the real setup (cuML RF/SVM, XGBoost and the MLP on CUDA,
TabPFN on CUDA, KNN on scikit-learn CPU — decision #24).
device="cpu" swaps in CPU versions (scikit-learn RF/SVM, CPU XGBoost/MLP/
TabPFN). It exists only for testing the pipeline without a GPU; never use
CPU results in the report.
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config as C


def _to_float32(X):
    return np.asarray(X, dtype=np.float32)


def _scaled(step_name: str, model) -> Pipeline:
    """Standardize (mean 0, std 1; fit on the training fold only) then the
    model — used for SVM, ANN and KNN, as in the paper."""
    return Pipeline([("scale", StandardScaler()),
                     ("f32", FunctionTransformer(_to_float32)),
                     (step_name, model)])


# ---------------------------------------------------------------- per model
def _rf(n_features, device, seed):
    grid = {"max_features": [m for m in C.GRIDS["rf"]["max_features"]
                             if m <= n_features]}
    if device == "gpu":
        from cuml.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=C.RF_N_TREES,
                                     max_depth=C.RF_MAX_DEPTH,
                                     random_state=seed)
    else:
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=C.RF_N_TREES,
                                     max_depth=C.RF_MAX_DEPTH,
                                     random_state=seed, n_jobs=-1)
    return est, grid


def _xgboost(n_features, device, seed):
    from xgboost import XGBClassifier
    est = XGBClassifier(learning_rate=C.XGB_LEARNING_RATE,
                        tree_method="hist",
                        device="cuda" if device == "gpu" else "cpu",
                        random_state=seed, n_jobs=1 if device == "gpu" else -1)
    return est, dict(C.GRIDS["xgboost"])


def _svm(n_features, device, seed):
    if device == "gpu":
        from cuml.svm import SVC
        svc = SVC(kernel="rbf")
    else:
        from sklearn.svm import SVC
        svc = SVC(kernel="rbf", random_state=seed)
    return _scaled("svc", svc), dict(C.GRIDS["svm"])


def _ann(n_features, device, seed):
    import torch
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit

    torch.manual_seed(seed)
    net = NeuralNetClassifier(
        MLP,
        module__d_in=n_features,
        module__hidden=(128,),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        optimizer__momentum=0.9,
        lr=C.ANN_LR,
        max_epochs=C.ANN_MAX_EPOCHS,
        batch_size=C.ANN_BATCH_SIZE,
        train_split=ValidSplit(C.ANN_VALID_FRAC, stratified=True,
                               random_state=seed),
        callbacks=[EarlyStopping(monitor="valid_loss",
                                 patience=C.ANN_PATIENCE, load_best=True)],
        device="cuda" if device == "gpu" else "cpu",
        iterator_train__shuffle=True,
        verbose=0,
    )
    return _scaled("net", net), dict(C.GRIDS["ann"])


def _knn(n_features, device, seed):
    from sklearn.neighbors import KNeighborsClassifier
    return (_scaled("knn", KNeighborsClassifier(metric="euclidean")),
            dict(C.GRIDS["knn"]))


def _tabpfn(n_features, device, seed):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    est = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cuda" if device == "gpu" else "cpu")
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est, {}


FACTORIES = {"rf": _rf, "xgboost": _xgboost, "svm": _svm, "ann": _ann,
             "knn": _knn, "tabpfn": _tabpfn}


def build(name: str, n_features: int, device: str = "gpu", seed: int = 0):
    if device == "gpu" and name in ("rf", "svm"):
        import cuml
        cuml.set_global_output_type("numpy")  # predictions come back as NumPy
    return FACTORIES[name](n_features, device, seed)


# ---------------------------------------------------------------- MLP module
try:
    import torch

    class MLP(torch.nn.Module):
        """Fully connected net: ReLU hidden layers, 2 outputs (logits)."""

        def __init__(self, d_in: int, hidden=(128,)):
            super().__init__()
            layers, prev = [], d_in
            for h in hidden:
                layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
                prev = h
            layers.append(torch.nn.Linear(prev, 2))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
except ImportError:  # torch only needed for the ANN
    MLP = None
