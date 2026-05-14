#!/usr/bin/env python3
# script: fit_and_export_gemf_panels.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gemf.workflows import fit_or_load_gemf, infer_consistent_scores, load_npz_dict


def norm_sid(x):
    sid = str(x).strip().replace("sub-", "")
    return sid.zfill(6) if sid.isdigit() else sid


def parse_str_list(s):
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_phi_matrix(basis_npz, K):
    z = np.load(basis_npz, allow_pickle=True)

    if "Phi" in z.files:
        Phi = z["Phi"]
    elif "phi" in z.files:
        Phi = z["phi"]
    elif "Phi_lh" in z.files and "Phi_rh" in z.files:
        Phi = np.vstack([z["Phi_lh"], z["Phi_rh"]])
    elif "phi_lh" in z.files and "phi_rh" in z.files:
        Phi = np.vstack([z["phi_lh"], z["phi_rh"]])
    else:
        raise RuntimeError(
            f"Could not find a cortical basis matrix in {basis_npz}. "
            "Expected Phi, phi, Phi_lh/Phi_rh, or phi_lh/phi_rh."
        )

    if Phi.shape[1] < K:
        raise RuntimeError(
            f"Basis matrix has {Phi.shape[1]} columns, but K={K} was requested."
        )

    return np.asarray(Phi[:, :K])


def leading_mode(M):
    M = (M + M.T) / 2.0
    evals, evecs = np.linalg.eigh(M)

    idx = int(np.argmax(np.abs(evals)))
    val = float(evals[idx])
    vec = np.asarray(evecs[:, idx]).copy()

    if val < 0:
        vec *= -1.0

    return vec, val


def score_column_names(R, Ls, Lf, Le):
    cols = [f"z{r + 1}" for r in range(R)]
    cols += [f"us{j + 1}" for j in range(Ls)]
    cols += [f"uf{j + 1}" for j in range(Lf)]
    cols += [f"ue{j + 1}" for j in range(Le)]
    return cols


def orient_shared_factors(fit, Zhat, condition_names=None, prefer_condition="EO", age=None):
    """
    Choose a stable sign convention for the shared factors.

    Preference order:
    1. Make the preferred EEG condition loading positive.
    2. If unavailable, make the age correlation positive.
    3. If unavailable, make the largest EEG spatial loading positive.
    """
    R = fit["A_shared"].shape[0]
    signs = np.ones(R)

    cond_idx = None
    if condition_names is not None:
        names = [str(x).strip().upper() for x in condition_names]
        target = prefer_condition.strip().upper()
        if target in names:
            cond_idx = names.index(target)

    for r in range(R):
        if cond_idx is not None:
            val = float(fit["c_shared"][r, cond_idx])
            if np.isfinite(val) and abs(val) > 1e-12:
                signs[r] = 1.0 if val >= 0 else -1.0
                continue

        if age is not None:
            ok = np.isfinite(age) & np.isfinite(Zhat[:, r])
            if ok.sum() >= 3:
                corr = np.corrcoef(Zhat[ok, r], age[ok])[0, 1]
                if np.isfinite(corr) and abs(corr) > 1e-10:
                    signs[r] = 1.0 if corr >= 0 else -1.0
                    continue

        j = int(np.argmax(np.abs(fit["a_shared"][r])))
        signs[r] = 1.0 if fit["a_shared"][r, j] >= 0 else -1.0

    aligned = {
        "Zhat": np.asarray(Zhat).copy() * signs[None, :],
        "A_shared": np.asarray(fit["A_shared"]).copy() * signs[:, None, None],
        "M_shared": np.asarray(fit["M_shared"]).copy() * signs[:, None, None],
        "a_shared": np.asarray(fit["a_shared"]).copy() * signs[:, None],
        "b_shared": np.asarray(fit["b_shared"]).copy() * signs[:, None],
        "c_shared": np.asarray(fit["c_shared"]).copy() * signs[:, None],
    }

    return signs, aligned


def save_scores(path, subject_ids, scores, columns, age=None, male=None):
    df = pd.DataFrame(scores, columns=columns)
    df.insert(0, "subject_id", subject_ids)

    if age is not None:
        df["age"] = age
    if male is not None:
        df["male"] = male

    df.round(3).to_csv(path, index=False)
    return df


def add_metadata(df, metadata_csv, subject_col, extra_cols, out_file):
    meta = pd.read_csv(metadata_csv)

    if subject_col not in meta.columns:
        raise RuntimeError(f"{metadata_csv} is missing subject column {subject_col}")

    keep_cols = [subject_col] + [c for c in parse_str_list(extra_cols) if c in meta.columns]

    meta = (
        meta[keep_cols]
        .assign(**{subject_col: meta[subject_col].map(norm_sid)})
        .drop_duplicates(subset=[subject_col])
        .rename(columns={subject_col: "subject_id"})
    )

    df.merge(meta, on="subject_id", how="left").round(3).to_csv(out_file, index=False)


def shared_factor_summary(Zhat, age=None):
    rows = []

    for r in range(Zhat.shape[1]):
        z = Zhat[:, r]
        row = {
            "factor": f"z{r + 1}",
            "mean": float(np.mean(z)),
            "sd": float(np.std(z, ddof=1)),
        }

        if age is not None:
            ok = np.isfinite(age) & np.isfinite(z)
            row["corr_age"] = (
                float(np.corrcoef(z[ok], age[ok])[0, 1]) if ok.sum() >= 3 else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def make_cortical_maps(Phi, aligned, R, q):
    V = Phi.shape[0]

    smri_maps = np.zeros((R, q, V), dtype=np.float32)
    eeg_maps = np.zeros((R, V), dtype=np.float32)
    fmri_diag = np.zeros((R, Phi.shape[1]), dtype=np.float32)
    fmri_mode = np.zeros((R, Phi.shape[1]), dtype=np.float32)
    fmri_eval = np.zeros(R, dtype=np.float32)
    fmri_map = np.zeros((R, V), dtype=np.float32)

    for r in range(R):
        for j in range(q):
            smri_maps[r, j] = Phi @ aligned["A_shared"][r, :, j]

        eeg_maps[r] = Phi @ aligned["a_shared"][r]

        M = aligned["M_shared"][r]
        fmri_diag[r] = np.diag(M)

        vec, val = leading_mode(M)
        fmri_mode[r] = vec
        fmri_eval[r] = val
        fmri_map[r] = Phi @ (val * vec)

    return {
        "smri_cortex_maps": smri_maps,
        "eeg_cortex_maps": eeg_maps,
        "fmri_eigenmode_loadings": fmri_diag,
        "fmri_mode_weights": fmri_mode,
        "fmri_eigenvalues": fmri_eval,
        "fmri_cortex_maps": fmri_map,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit full-data GEMF and export plot-ready panel objects."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--Ls", type=int, default=1)
    parser.add_argument("--Lf", type=int, default=1)
    parser.add_argument("--Le", type=int, default=1)

    parser.add_argument("--w_s", type=float, default=1.0)
    parser.add_argument("--w_f", type=float, default=1.0)
    parser.add_argument("--w_e", type=float, default=1.0)

    parser.add_argument("--lambda_s", type=float, default=1e-2)
    parser.add_argument("--lambda_f", type=float, default=1e-2)
    parser.add_argument("--lambda_e", type=float, default=1e-2)
    parser.add_argument("--zeta_e", type=float, default=1e-2)

    parser.add_argument("--specific_ridge", type=float, default=1e-6)
    parser.add_argument("--score_ridge", type=float, default=1e-1)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--basis_npz", default="")
    parser.add_argument("--metadata_csv", default="")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--extra_cols", default="age_mid,male,LPS_1")
    parser.add_argument("--resume_fit", action="store_true")
    parser.add_argument("--prefer_condition", default="EO")

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_npz_dict(args.data)

    subject_ids = np.array([norm_sid(x) for x in raw["subject_ids"]], dtype="U6")
    K = int(np.asarray(raw["K"]).item())
    q = int(np.asarray(raw["q"]).item())
    F = int(np.asarray(raw["F"]).item())
    C = int(np.asarray(raw["C"]).item())

    age = np.asarray(raw["age"], dtype=float) if "age" in raw else None
    male = np.asarray(raw["male"], dtype=float) if "male" in raw else None

    smri_measures = raw.get("smri_measures", np.array([f"measure{j + 1}" for j in range(q)]))
    eeg_freq_centers = raw.get("eeg_freq_centers", np.arange(F))
    eeg_freq_bands = raw.get("eeg_freq_bands", np.array([], dtype=object))
    condition_names = raw.get("condition_names", np.array([f"cond{c + 1}" for c in range(C)]))

    fit, _ = fit_or_load_gemf(
        data_source=args.data,
        fit_dir=outdir,
        R=args.R,
        Ls=args.Ls,
        Lf=args.Lf,
        Le=args.Le,
        w_s=args.w_s,
        w_f=args.w_f,
        w_e=args.w_e,
        lambda_s=args.lambda_s,
        lambda_f=args.lambda_f,
        lambda_e=args.lambda_e,
        zeta_e=args.zeta_e,
        specific_ridge=args.specific_ridge,
        score_ridge=args.score_ridge,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        resume=args.resume_fit,
        quiet=args.quiet,
    )

    scores = infer_consistent_scores(
        fit=fit,
        data_source=args.data,
        w_s=args.w_s,
        w_f=args.w_f,
        w_e=args.w_e,
        score_ridge=args.score_ridge,
    )

    Zhat = scores["Zhat"]
    score_full = scores["score"]

    fit["Zhat_train"] = Zhat
    fit["score_train"] = score_full
    fit["score_ridge"] = np.array(args.score_ridge, dtype=float)
    np.savez_compressed(outdir / "fit_results.npz", **fit)

    factor_signs, aligned = orient_shared_factors(
        fit,
        Zhat,
        condition_names=[str(x) for x in condition_names],
        prefer_condition=args.prefer_condition,
        age=age,
    )

    score_aligned = np.asarray(score_full).copy()
    score_aligned[:, : args.R] *= factor_signs[None, :]

    shared_cols = [f"z{r + 1}" for r in range(args.R)]
    full_cols = score_column_names(args.R, args.Ls, args.Lf, args.Le)

    df_shared = save_scores(
        outdir / "shared_scores.csv",
        subject_ids,
        aligned["Zhat"],
        shared_cols,
        age=age,
        male=male,
    )

    df_full = save_scores(
        outdir / "full_scores.csv",
        subject_ids,
        score_aligned,
        full_cols,
        age=age,
        male=male,
    )

    if args.metadata_csv:
        add_metadata(
            df_shared,
            args.metadata_csv,
            args.subject_col,
            args.extra_cols,
            outdir / "shared_scores_with_meta.csv",
        )

        add_metadata(
            df_full,
            args.metadata_csv,
            args.subject_col,
            args.extra_cols,
            outdir / "full_scores_with_meta.csv",
        )

    shared_factor_summary(aligned["Zhat"], age).round(3).to_csv(
        outdir / "shared_factor_summary.csv",
        index=False,
    )

    panel_objects = {
        "subject_ids": subject_ids,
        "shared_scores": aligned["Zhat"],
        "full_scores": score_aligned,
        "factor_signs": factor_signs,
        "A_shared": aligned["A_shared"],
        "M_shared": aligned["M_shared"],
        "a_shared": aligned["a_shared"],
        "b_shared": aligned["b_shared"],
        "c_shared": aligned["c_shared"],
        "smri_measures": smri_measures,
        "eeg_freq_centers": eeg_freq_centers,
        "eeg_freq_bands": eeg_freq_bands,
        "condition_names": condition_names,
        "R": np.array(args.R, dtype=int),
        "Ls": np.array(args.Ls, dtype=int),
        "Lf": np.array(args.Lf, dtype=int),
        "Le": np.array(args.Le, dtype=int),
        "K": np.array(K, dtype=int),
        "q": np.array(q, dtype=int),
        "F": np.array(F, dtype=int),
        "C": np.array(C, dtype=int),
        "prefer_condition": np.array(args.prefer_condition),
    }

    if age is not None:
        panel_objects["age"] = age
    if male is not None:
        panel_objects["male"] = male
    if "lambda_vals" in raw:
        panel_objects["lambda_vals"] = np.asarray(raw["lambda_vals"])

    if args.basis_npz:
        Phi = load_phi_matrix(args.basis_npz, K)
        panel_objects.update(make_cortical_maps(Phi, aligned, args.R, q))

    np.savez_compressed(outdir / "panel_objects.npz", **panel_objects)

    summary = {
        "model_name": "GEMF",
        "R": args.R,
        "Ls": args.Ls,
        "Lf": args.Lf,
        "Le": args.Le,
        "score_ridge": args.score_ridge,
        "n_subjects": int(len(subject_ids)),
        "prefer_condition": args.prefer_condition,
    }

    with open(outdir / "panel_export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name in [
        "fit_results.npz",
        "shared_scores.csv",
        "full_scores.csv",
        "shared_factor_summary.csv",
        "panel_objects.npz",
        "panel_export_summary.json",
    ]:
        print("Wrote:", outdir / name)


if __name__ == "__main__":
    main()