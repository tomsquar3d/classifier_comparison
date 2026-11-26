"""Quick check that every model library imports and runs on the GPU.

Fits each of the 6 classifiers on a small random dataset and prints
OK / FAIL per model, plus library versions and the GPU name. Takes about
a minute (TabPFN downloads its weights the first time).

Usage (inside the DSMLP container, with the cogs118a env active):
    python src/smoke_test.py
"""
from __future__ import annotations

import platform
import subprocess
import time
import traceback

import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 20)).astype(np.float32)
y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.5, size=2000) > 1).astype(np.int64)
Xtr, Xte, ytr, yte = X[:1500], X[1500:], y[:1500], y[1500:]


def gpu_info() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ["numpy", "sklearn", "cuml", "xgboost", "torch", "skorch", "tabpfn"]:
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"IMPORT FAILED ({type(e).__name__})"
    try:
        import torch
        out["torch.cuda"] = f"{torch.version.cuda}, available={torch.cuda.is_available()}"
    except Exception:  # noqa: BLE001
        pass
    return out


def to_numpy(a):
    """cuML can return cupy/cudf; bring predictions back as NumPy."""
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    if hasattr(a, "get"):
        a = a.get()
    return np.asarray(a).ravel()


def rf():
    from cuml.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, max_depth=16, max_features=4,
                                  random_state=0)


def xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, max_depth=4, device="cuda",
                         tree_method="hist")


def svm():
    from cuml.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), SVC(C=1.0, gamma=0.05))


def ann():
    import torch
    from skorch import NeuralNetClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class MLP(torch.nn.Module):
        def __init__(self, d_in=20, h=128):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_in, h), torch.nn.ReLU(),
                torch.nn.Linear(h, h), torch.nn.ReLU(),
                torch.nn.Linear(h, 2))

        def forward(self, x):
            return self.net(x)

    net = NeuralNetClassifier(MLP, criterion=torch.nn.CrossEntropyLoss,
                              optimizer=torch.optim.SGD, optimizer__momentum=0.9,
                              lr=0.01, max_epochs=20, device="cuda", verbose=0)
    return make_pipeline(StandardScaler(), net)


def knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=15, weights="distance"))


def tabpfn():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2,
                                                       device="cuda")


MODELS = {"rf (cuML)": rf, "xgboost (cuda)": xgb, "svm (cuML)": svm,
          "ann (skorch, cuda)": ann, "knn (sklearn)": knn, "tabpfn (v2)": tabpfn}


def main() -> None:
    print("GPU:", gpu_info())
    for k, v in versions().items():
        print(f"  {k:12s} {v}")
    print()
    n_ok = 0
    for name, make in MODELS.items():
        t0 = time.perf_counter()
        try:
            model = make()
            ytr_fit = ytr if not name.startswith("ann") else ytr.astype(np.int64)
            model.fit(Xtr, ytr_fit)
            acc = float((to_numpy(model.predict(Xte)) == yte).mean())
            print(f"OK    {name:20s} test acc {acc:.3f}  ({time.perf_counter() - t0:.1f}s)")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name:20s} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\n{n_ok}/{len(MODELS)} models passed")


if __name__ == "__main__":
    main()
