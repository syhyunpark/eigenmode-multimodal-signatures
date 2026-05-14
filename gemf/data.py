from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MultiViewData:
    """
    Container for the estimator-scale multimodal arrays.

    Shapes
    ------
    S_tilde      : (n, K, q)
    Sigma_tilde  : (n, K, K)
    E            : (n, K, F, C)
    lambda_vals  : (K,)
    age          : (n,) optional
    subject_ids  : (n,) optional
    """
    S_tilde: np.ndarray
    Sigma_tilde: np.ndarray
    E: np.ndarray
    lambda_vals: np.ndarray
    age: Optional[np.ndarray] = None
    subject_ids: Optional[np.ndarray] = None

    def validate(self) -> None:
        if self.S_tilde.ndim != 3:
            raise ValueError(f"S_tilde must be 3D, got shape {self.S_tilde.shape}")
        if self.Sigma_tilde.ndim != 3:
            raise ValueError(f"Sigma_tilde must be 3D, got shape {self.Sigma_tilde.shape}")
        if self.E.ndim != 4:
            raise ValueError(f"E must be 4D, got shape {self.E.shape}")
        if self.lambda_vals.ndim != 1:
            raise ValueError("lambda_vals must be 1D")

        n_s, K_s, q = self.S_tilde.shape
        n_f, K_f1, K_f2 = self.Sigma_tilde.shape
        n_e, K_e, F, C = self.E.shape

        if not (n_s == n_f == n_e):
            raise ValueError("Number of subjects must match across modalities")
        if not (K_s == K_f1 == K_f2 == K_e == self.lambda_vals.shape[0]):
            raise ValueError("Spatial mode dimension K must match across all arrays")

        if self.age is not None and self.age.shape[0] != n_s:
            raise ValueError("age has incompatible length")
        if self.subject_ids is not None and self.subject_ids.shape[0] != n_s:
            raise ValueError("subject_ids has incompatible length")

    @property
    def n(self) -> int:
        return self.S_tilde.shape[0]

    @property
    def K(self) -> int:
        return self.S_tilde.shape[1]

    @property
    def q(self) -> int:
        return self.S_tilde.shape[2]

    @property
    def F(self) -> int:
        return self.E.shape[2]

    @property
    def C(self) -> int:
        return self.E.shape[3]

    def subset(self, indices: np.ndarray) -> "MultiViewData":
        age = None if self.age is None else self.age[indices]
        subject_ids = None if self.subject_ids is None else self.subject_ids[indices]
        return MultiViewData(
            S_tilde=self.S_tilde[indices],
            Sigma_tilde=self.Sigma_tilde[indices],
            E=self.E[indices],
            lambda_vals=self.lambda_vals.copy(),
            age=age,
            subject_ids=subject_ids,
        )


def load_multiview_npz(path: str | Path) -> MultiViewData:
    """
    Load the standardized multiview arrays from a .npz file produced by
    the simulation script (or by a later real-data preparation script).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    arr = np.load(path, allow_pickle=True)

    required = ["S_tilde", "Sigma_tilde", "E", "lambda_vals"]
    missing = [k for k in required if k not in arr]
    if missing:
        raise KeyError(f"Missing required arrays in {path}: {missing}")

    age = arr["age"] if "age" in arr else None
    subject_ids = arr["subject_ids"] if "subject_ids" in arr else None

    data = MultiViewData(
        S_tilde=np.asarray(arr["S_tilde"], dtype=float),
        Sigma_tilde=np.asarray(arr["Sigma_tilde"], dtype=float),
        E=np.asarray(arr["E"], dtype=float),
        lambda_vals=np.asarray(arr["lambda_vals"], dtype=float),
        age=None if age is None else np.asarray(age, dtype=float),
        subject_ids=None if subject_ids is None else np.asarray(subject_ids),
    )
    data.validate()
    return data