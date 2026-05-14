# gemf/workflows.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gemf.data import load_multiview_npz
from gemf.fixed_rank_model import FixedRankGMHFConfig as FixedRankGEMFConfig
from gemf.fixed_rank_model import FixedRankGMHF as FixedRankGEMF
from gemf.score_inference import (
    infer_shared_scores,
    infer_full_scores,
    build_shared_tensor_cols,
    build_specific_tensor_cols,
    pack_subject_multiview,
)


def load_npz_dict(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def save_npz_dict(path: str | Path, d: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def subset_npz_dict(d: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    n = int(np.asarray(d["n"]).item()) if "n" in d else len(d["subject_ids"])
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[k] = arr[idx]
        else:
            out[k] = arr
    out["n"] = np.array(len(idx), dtype=np.int32)
    return out


def load_data_object(data_source: Any):
    if isinstance(data_source, (str, Path)):
        return load_multiview_npz(str(data_source))
    return data_source


def scale_views_by_train(
    train_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    mode: str = "entry_rms",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Fold-specific view scaling.

    mode:
      - none
      - fro_mean
      - entry_rms
    """
    tr = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in train_dict.items()}
    te = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in test_dict.items()}

    if mode == "none":
        return tr, te, {"s": 1.0, "f": 1.0, "e": 1.0}

    def mean_scale(X: np.ndarray, entry_rms: bool) -> float:
        vals = []
        p = np.prod(X.shape[1:])
        for i in range(X.shape[0]):
            fro = np.linalg.norm(X[i].reshape(-1))
            if entry_rms:
                fro = fro / np.sqrt(p)
            vals.append(fro)
        sc = float(np.mean(vals))
        return max(sc, 1e-8)

    entry_rms = (mode == "entry_rms")
    s_sc = mean_scale(tr["S_tilde"], entry_rms)
    f_sc = mean_scale(tr["Sigma_tilde"], entry_rms)
    e_sc = mean_scale(tr["E"], entry_rms)

    tr["S_tilde"] = tr["S_tilde"] / s_sc
    tr["Sigma_tilde"] = tr["Sigma_tilde"] / f_sc
    tr["E"] = tr["E"] / e_sc

    te["S_tilde"] = te["S_tilde"] / s_sc
    te["Sigma_tilde"] = te["Sigma_tilde"] / f_sc
    te["E"] = te["E"] / e_sc

    return tr, te, {"s": s_sc, "f": f_sc, "e": e_sc}


def fit_gemf_model(
    data_source: Any,
    *,
    R: int,
    Ls: int,
    Lf: int,
    Le: int,
    w_s: float,
    w_f: float,
    w_e: float,
    lambda_s: float,
    lambda_f: float,
    lambda_e: float,
    zeta_e: float,
    specific_ridge: float,
    score_ridge: float,
    max_iter: int,
    tol: float,
    seed: int,
    quiet: bool = False,
):
    """
    Fit GEMF from a data object or .npz path.
    Returns (fit_dict, data_obj, model).
    """
    data = load_data_object(data_source)

    config = FixedRankGEMFConfig(
        R=R,
        Ls=Ls,
        Lf=Lf,
        Le=Le,
        w_s=w_s,
        w_f=w_f,
        w_e=w_e,
        lambda_s=lambda_s,
        lambda_f=lambda_f,
        lambda_e=lambda_e,
        zeta_e=zeta_e,
        specific_ridge=specific_ridge,
        score_ridge=score_ridge,
        prefit_shared_iters=0,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        verbose=not quiet,
    )

    model = FixedRankGEMF(config=config)
    model.fit(data)
    fit = model.save_dict(data=data)
    return fit, data, model


def fit_or_load_gemf(
    data_source: Any,
    fit_dir: str | Path,
    *,
    R: int,
    Ls: int,
    Lf: int,
    Le: int,
    w_s: float,
    w_f: float,
    w_e: float,
    lambda_s: float,
    lambda_f: float,
    lambda_e: float,
    zeta_e: float,
    specific_ridge: float,
    score_ridge: float,
    max_iter: int,
    tol: float,
    seed: int,
    resume: bool = True,
    quiet: bool = False,
):
    """
    Fit GEMF or load a cached fit_results.npz from fit_dir.
    Returns (fit_dict, data_obj).
    """
    fit_dir = Path(fit_dir)
    fit_dir.mkdir(parents=True, exist_ok=True)
    fit_npz = fit_dir / "fit_results.npz"

    data = load_data_object(data_source)

    if resume and fit_npz.exists():
        fit = load_npz_dict(fit_npz)
        return fit, data

    fit, data, _ = fit_gemf_model(
        data_source=data,
        R=R,
        Ls=Ls,
        Lf=Lf,
        Le=Le,
        w_s=w_s,
        w_f=w_f,
        w_e=w_e,
        lambda_s=lambda_s,
        lambda_f=lambda_f,
        lambda_e=lambda_e,
        zeta_e=zeta_e,
        specific_ridge=specific_ridge,
        score_ridge=score_ridge,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        quiet=quiet,
    )
    np.savez_compressed(fit_npz, **fit)
    return fit, data


def infer_consistent_scores(
    fit: Dict[str, np.ndarray],
    data_source: Any,
    *,
    w_s: float,
    w_f: float,
    w_e: float,
    score_ridge: float,
) -> Dict[str, np.ndarray]:
    """
    Compute the two canonical downstream score representations:
      - Zhat : shared-only inferred scores
      - score: full inferred downstream score vector [Z, U_s, U_f, U_e]
    """
    data = load_data_object(data_source)

    Zhat = infer_shared_scores(
        fit=fit,
        data_like=data,
        w_s=w_s,
        w_f=w_f,
        w_e=w_e,
        ridge=score_ridge,
    )

    score = infer_full_scores(
        fit=fit,
        data_like=data,
        w_s=w_s,
        w_f=w_f,
        w_e=w_e,
        ridge=score_ridge,
    )

    return {
        "Zhat": Zhat,
        "score": score,
        "score_ridge": np.array(score_ridge, dtype=float),
    }


def augment_fit_with_consistent_scores(
    fit: Dict[str, np.ndarray],
    data_source: Any,
    *,
    w_s: float,
    w_f: float,
    w_e: float,
    score_ridge: float,
) -> Dict[str, np.ndarray]:
    """
    Return a copy of fit augmented with:
      - Zhat_train
      - score_train
      - score_ridge
    """
    out = dict(fit)
    score_dict = infer_consistent_scores(
        fit=fit,
        data_source=data_source,
        w_s=w_s,
        w_f=w_f,
        w_e=w_e,
        score_ridge=score_ridge,
    )
    out["Zhat_train"] = score_dict["Zhat"]
    out["score_train"] = score_dict["score"]
    out["score_ridge"] = score_dict["score_ridge"]
    return out


def build_full_dictionary(
    fit: Dict[str, np.ndarray],
    *,
    w_s: float,
    w_f: float,
    w_e: float,
) -> np.ndarray:
    X_shared = build_shared_tensor_cols(fit, w_s, w_f, w_e)
    X_spec = build_specific_tensor_cols(fit, w_s, w_f, w_e)
    if X_spec is None:
        return X_shared
    return np.column_stack([X_shared, X_spec])


def normalized_reconstruction_error(
    fit: Dict[str, np.ndarray],
    data_source: Any,
    *,
    w_s: float,
    w_f: float,
    w_e: float,
    score_ridge: float,
) -> float:
    """
    Normalized multiview reconstruction error:
      ||Y - Yhat||^2 / ||Y||^2
    using the full inferred downstream score representation.

    Accepts either:
      - a GEMF data object, or
      - a dict with keys S_tilde, Sigma_tilde, E
    """
    data = load_data_object(data_source)

    if isinstance(data, dict):
        S_tilde = np.asarray(data["S_tilde"])
        Sigma_tilde = np.asarray(data["Sigma_tilde"])
        E = np.asarray(data["E"])
    else:
        S_tilde = np.asarray(data.S_tilde)
        Sigma_tilde = np.asarray(data.Sigma_tilde)
        E = np.asarray(data.E)

    X = build_full_dictionary(fit, w_s=w_s, w_f=w_f, w_e=w_e)
    score = infer_full_scores(
        fit=fit,
        data_like=data,
        w_s=w_s,
        w_f=w_f,
        w_e=w_e,
        ridge=score_ridge,
    )

    rss = 0.0
    tss = 0.0
    n = S_tilde.shape[0]

    for i in range(n):
        y = pack_subject_multiview(
            S_tilde[i],
            Sigma_tilde[i],
            E[i],
            w_s=w_s,
            w_f=w_f,
            w_e=w_e,
        )
        yhat = X @ score[i]
        rss += float(np.sum((y - yhat) ** 2))
        tss += float(np.sum(y ** 2))

    return rss / max(tss, 1e-12)