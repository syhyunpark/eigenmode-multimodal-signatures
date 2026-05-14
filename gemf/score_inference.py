# gemf/score_inference.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def _as_arrays(data_like: Any) -> Dict[str, np.ndarray]:
    if isinstance(data_like, dict):
        return {
            "S_tilde": np.asarray(data_like["S_tilde"]),
            "Sigma_tilde": np.asarray(data_like["Sigma_tilde"]),
            "E": np.asarray(data_like["E"]),
        }
    return {
        "S_tilde": np.asarray(data_like.S_tilde),
        "Sigma_tilde": np.asarray(data_like.Sigma_tilde),
        "E": np.asarray(data_like.E),
    }


def build_shared_tensor_cols(
    fit: Dict[str, np.ndarray],
    w_s: float,
    w_f: float,
    w_e: float,
) -> np.ndarray:
    A = fit["A_shared"]
    M = fit["M_shared"]
    a = fit["a_shared"]
    b = fit["b_shared"]
    c = fit["c_shared"]

    cols = []
    for r in range(A.shape[0]):
        e_col = np.einsum("k,f,c->kfc", a[r], b[r], c[r]).reshape(-1)
        col = np.concatenate([
            np.sqrt(w_s) * A[r].reshape(-1),
            np.sqrt(w_f) * M[r].reshape(-1),
            np.sqrt(w_e) * e_col,
        ])
        cols.append(col)
    return np.column_stack(cols)


def build_specific_tensor_cols(
    fit: Dict[str, np.ndarray],
    w_s: float,
    w_f: float,
    w_e: float,
) -> np.ndarray | None:
    cols = []

    A0 = fit["A_shared"][0]
    M0 = fit["M_shared"][0]
    E0 = np.einsum("k,f,c->kfc", fit["a_shared"][0], fit["b_shared"][0], fit["c_shared"][0])

    if "A_specific_s" in fit and fit["A_specific_s"].size > 0:
        for l in range(fit["A_specific_s"].shape[0]):
            cols.append(np.concatenate([
                np.sqrt(w_s) * fit["A_specific_s"][l].reshape(-1),
                np.sqrt(w_f) * np.zeros_like(M0).reshape(-1),
                np.sqrt(w_e) * np.zeros_like(E0).reshape(-1),
            ]))

    if "M_specific_f" in fit and fit["M_specific_f"].size > 0:
        for l in range(fit["M_specific_f"].shape[0]):
            cols.append(np.concatenate([
                np.sqrt(w_s) * np.zeros_like(A0).reshape(-1),
                np.sqrt(w_f) * fit["M_specific_f"][l].reshape(-1),
                np.sqrt(w_e) * np.zeros_like(E0).reshape(-1),
            ]))

    if "a_specific_e" in fit and fit["a_specific_e"].size > 0:
        for l in range(fit["a_specific_e"].shape[0]):
            e_col = np.einsum(
                "k,f,c->kfc",
                fit["a_specific_e"][l],
                fit["b_specific_e"][l],
                fit["c_specific_e"][l],
            ).reshape(-1)
            cols.append(np.concatenate([
                np.sqrt(w_s) * np.zeros_like(A0).reshape(-1),
                np.sqrt(w_f) * np.zeros_like(M0).reshape(-1),
                np.sqrt(w_e) * e_col,
            ]))

    if len(cols) == 0:
        return None
    return np.column_stack(cols)


def pack_subject_multiview(
    S_tilde: np.ndarray,
    Sigma_tilde: np.ndarray,
    E: np.ndarray,
    w_s: float,
    w_f: float,
    w_e: float,
) -> np.ndarray:
    return np.concatenate([
        np.sqrt(w_s) * S_tilde.reshape(-1),
        np.sqrt(w_f) * Sigma_tilde.reshape(-1),
        np.sqrt(w_e) * E.reshape(-1),
    ])


def infer_shared_scores(
    fit: Dict[str, np.ndarray],
    data_like: Any,
    w_s: float,
    w_f: float,
    w_e: float,
    ridge: float = 0.1,
) -> np.ndarray:
    d = _as_arrays(data_like)
    X_shared = build_shared_tensor_cols(fit, w_s, w_f, w_e)

    XtX = X_shared.T @ X_shared + ridge * np.eye(X_shared.shape[1])
    solver = np.linalg.solve(XtX, X_shared.T)

    n = d["S_tilde"].shape[0]
    Zhat = np.zeros((n, X_shared.shape[1]), dtype=np.float32)

    for i in range(n):
        y = pack_subject_multiview(
            d["S_tilde"][i],
            d["Sigma_tilde"][i],
            d["E"][i],
            w_s=w_s, w_f=w_f, w_e=w_e,
        )
        Zhat[i] = solver @ y

    return Zhat


def infer_full_scores(
    fit: Dict[str, np.ndarray],
    data_like: Any,
    w_s: float,
    w_f: float,
    w_e: float,
    ridge: float = 0.1,
) -> np.ndarray:
    d = _as_arrays(data_like)

    X_shared = build_shared_tensor_cols(fit, w_s, w_f, w_e)
    X_spec = build_specific_tensor_cols(fit, w_s, w_f, w_e)

    if X_spec is None:
        X = X_shared
    else:
        X = np.column_stack([X_shared, X_spec])

    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    solver = np.linalg.solve(XtX, X.T)

    n = d["S_tilde"].shape[0]
    score = np.zeros((n, X.shape[1]), dtype=np.float32)

    for i in range(n):
        y = pack_subject_multiview(
            d["S_tilde"][i],
            d["Sigma_tilde"][i],
            d["E"][i],
            w_s=w_s, w_f=w_f, w_e=w_e,
        )
        score[i] = solver @ y

    return score