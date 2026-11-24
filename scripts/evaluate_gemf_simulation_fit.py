#!/usr/bin/env python3
# script: evaluate_gemf_simulation_fit.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate GEMF recovery in simulated data.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def energy_r2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    den = float(np.sum(y_true ** 2))
    if den <= 1e-12:
        return np.nan
    return 1.0 - float(np.sum((y_true - y_hat) ** 2)) / den


def canonical_corrs(X_est: np.ndarray, X_true: np.ndarray) -> np.ndarray:
    if X_est is None or X_true is None or X_est.size == 0 or X_true.size == 0:
        return np.zeros(0)

    Xe = X_est - X_est.mean(axis=0, keepdims=True)
    Xt = X_true - X_true.mean(axis=0, keepdims=True)

    Qe, _ = np.linalg.qr(Xe)
    Qt, _ = np.linalg.qr(Xt)

    s = np.linalg.svd(Qe.T @ Qt, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def cc_mean(X_est: np.ndarray, X_true: np.ndarray) -> float:
    s = canonical_corrs(X_est, X_true)
    return float(np.mean(s)) if s.size else np.nan


def true_shared(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "S": np.einsum("nr,rkq->nkq", data["Z_true"], data["A_shared_true"]),
        "Sigma": np.einsum("nr,rkm->nkm", data["Z_true"], data["M_shared_true"]),
        "E": np.einsum(
            "nr,rk,rf,rc->nkfc",
            data["Z_true"],
            data["a_shared_true"],
            data["b_shared_true"],
            data["c_shared_true"],
        ),
    }


def fit_shared(fit: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "S": np.einsum("nr,rkq->nkq", fit["Z"], fit["A_shared"]),
        "Sigma": np.einsum("nr,rkm->nkm", fit["Z"], fit["M_shared"]),
        "E": np.einsum(
            "nr,rk,rf,rc->nkfc",
            fit["Z"],
            fit["a_shared"],
            fit["b_shared"],
            fit["c_shared"],
        ),
    }


def true_specific(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | None]:
    out = {"S": None, "Sigma": None, "E": None}

    if data["U_s_true"].size > 0:
        out["S"] = np.einsum("nl,lkq->nkq", data["U_s_true"], data["A_specific_true"])

    if data["U_f_true"].size > 0:
        out["Sigma"] = np.einsum("nl,lkm->nkm", data["U_f_true"], data["M_specific_true"])

    if data["U_e_true"].size > 0:
        out["E"] = np.einsum(
            "nl,lk,lf,lc->nkfc",
            data["U_e_true"],
            data["a_specific_true"],
            data["b_specific_true"],
            data["c_specific_true"],
        )

    return out


def fit_specific(fit: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | None]:
    out = {"S": None, "Sigma": None, "E": None}

    if "U_s" in fit and "A_specific_s" in fit and fit["U_s"].size > 0:
        out["S"] = np.einsum("nl,lkq->nkq", fit["U_s"], fit["A_specific_s"])

    if "U_f" in fit and "M_specific_f" in fit and fit["U_f"].size > 0:
        out["Sigma"] = np.einsum("nl,lkm->nkm", fit["U_f"], fit["M_specific_f"])

    if "U_e" in fit and "a_specific_e" in fit and fit["U_e"].size > 0:
        out["E"] = np.einsum(
            "nl,lk,lf,lc->nkfc",
            fit["U_e"],
            fit["a_specific_e"],
            fit["b_specific_e"],
            fit["c_specific_e"],
        )

    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data)
    fit = load_npz(args.fit)

    sh_true = true_shared(data)
    sh_fit = fit_shared(fit)

    sp_true = true_specific(data)
    sp_fit = fit_specific(fit)

    summary: Dict[str, Any] = {}

    # Use the inferred shared scores when available, because this is the
    # score definition used in the MPI-LEMON application. Component
    # reconstruction below still uses the internal fitted Z with fitted loadings.
    Z_for_score = fit["Zhat_train"] if "Zhat_train" in fit else fit["Z"]
    summary["Z_subspace_cc_mean"] = cc_mean(Z_for_score, data["Z_true"])

    summary["smri_shared_r2"] = energy_r2(sh_true["S"], sh_fit["S"])
    summary["fmri_shared_r2"] = energy_r2(sh_true["Sigma"], sh_fit["Sigma"])
    summary["eeg_shared_r2"] = energy_r2(sh_true["E"], sh_fit["E"])

    summary["U_s_subspace_cc_mean"] = cc_mean(fit.get("U_s", np.zeros((data["Z_true"].shape[0], 0))), data["U_s_true"])
    summary["U_f_subspace_cc_mean"] = cc_mean(fit.get("U_f", np.zeros((data["Z_true"].shape[0], 0))), data["U_f_true"])
    summary["U_e_subspace_cc_mean"] = cc_mean(fit.get("U_e", np.zeros((data["Z_true"].shape[0], 0))), data["U_e_true"])

    for key, label in [("S", "smri"), ("Sigma", "fmri"), ("E", "eeg")]:
        if sp_true[key] is None or sp_fit[key] is None:
            summary[f"{label}_specific_r2"] = np.nan
        else:
            summary[f"{label}_specific_r2"] = energy_r2(sp_true[key], sp_fit[key])

    summary["scenario"] = str(data["scenario"].item()) if "scenario" in data else ""
    summary["n"] = int(data["n"].item()) if "n" in data else int(data["S_tilde"].shape[0])
    summary["true_R"] = int(data["true_R"].item()) if "true_R" in data else int(data["Z_true"].shape[1])
    summary["true_Ls"] = int(data["true_Ls"].item()) if "true_Ls" in data else int(data["U_s_true"].shape[1])
    summary["true_Lf"] = int(data["true_Lf"].item()) if "true_Lf" in data else int(data["U_f_true"].shape[1])
    summary["true_Le"] = int(data["true_Le"].item()) if "true_Le" in data else int(data["U_e_true"].shape[1])

    with open(outdir / "simulation_recovery_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", outdir / "simulation_recovery_summary.json")


if __name__ == "__main__":
    main()