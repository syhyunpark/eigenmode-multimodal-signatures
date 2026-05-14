#!/usr/bin/env python3
# script: simulate_gemf_energy_data.py

import argparse
import json
from pathlib import Path

import numpy as np


SCENARIOS = {
    "shared_only": {
        "energy": (0.70, 0.00, 0.30),
        "specific_ranks": (0, 0, 0),
    },
    "shared_specific": {
        "energy": (0.45, 0.35, 0.20),
        "specific_ranks": (1, 1, 1),
    },
    "low_snr_specific": {
        "energy": (0.35, 0.05, 0.60),
        "specific_ranks": (1, 1, 1),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate GEMF data using target energy proportions."
    )

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS),
    )

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--F", type=int, default=16)
    parser.add_argument("--C", type=int, default=2)

    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--Ls", type=int, default=None)
    parser.add_argument("--Lf", type=int, default=None)
    parser.add_argument("--Le", type=int, default=None)

    parser.add_argument("--lambda-power", type=float, default=1.0)
    parser.add_argument("--age-mean", type=float, default=50.0)
    parser.add_argument("--age-sd", type=float, default=15.0)
    parser.add_argument("--age-noise-sd", type=float, default=4.0)

    return parser.parse_args()


def normalize(x, target_norm=1.0):
    norm = np.linalg.norm(x.ravel())
    if norm < 1e-12:
        return x.copy()
    return x * (target_norm / norm)


def symmetrize(x):
    return 0.5 * (x + x.T)


def sample_scores(n, r, rng, orthogonal_to=None):
    if r == 0:
        return np.zeros((n, 0), dtype=float)

    X = rng.normal(size=(n, r))

    if orthogonal_to is not None and orthogonal_to.size > 0:
        Q, _ = np.linalg.qr(orthogonal_to)
        X = X - Q @ (Q.T @ X)

    Q, _ = np.linalg.qr(X)
    return np.sqrt(n) * Q[:, :r]


def random_frequency_profile(F, rng):
    grid = np.linspace(0.0, 1.0, F)
    profile = np.zeros(F)

    for _ in range(int(rng.integers(1, 3))):
        center = rng.uniform(0.1, 0.9)
        width = rng.uniform(0.06, 0.18)
        height = rng.uniform(0.5, 1.5)

        profile += height * np.exp(-0.5 * ((grid - center) / width) ** 2)

    profile += 1e-4
    return normalize(profile)


def random_condition_profile(C, rng):
    return normalize(rng.normal(size=C))


def sample_smri_loadings(R, K, q, decay, rng):
    A = np.zeros((R, K, q))

    for r in range(R):
        A[r] = normalize(decay[:, None] * rng.normal(size=(K, q)))

    return A


def sample_fmri_loadings(R, K, decay, rng):
    M = np.zeros((R, K, K))
    weight = np.outer(decay, decay)

    for r in range(R):
        M[r] = symmetrize(normalize(weight * symmetrize(rng.normal(size=(K, K)))))

    return M


def sample_eeg_loadings(R, K, F, C, decay, rng):
    a = np.zeros((R, K))
    b = np.zeros((R, F))
    c = np.zeros((R, C))

    for r in range(R):
        a[r] = normalize(decay * rng.normal(size=K))
        b[r] = random_frequency_profile(F, rng)
        c[r] = random_condition_profile(C, rng)

    return a, b, c


def reconstruct_smri(Z, A):
    if Z.size == 0 or A.size == 0:
        return np.zeros((Z.shape[0], A.shape[1], A.shape[2]))
    return np.einsum("nr,rkq->nkq", Z, A)


def reconstruct_fmri(Z, M):
    if Z.size == 0 or M.size == 0:
        return np.zeros((Z.shape[0], M.shape[1], M.shape[2]))
    return np.einsum("nr,rkm->nkm", Z, M)


def reconstruct_eeg(Z, a, b, c):
    if Z.size == 0 or a.size == 0:
        return np.zeros((Z.shape[0], a.shape[1], b.shape[1], c.shape[1]))
    return np.einsum("nr,rk,rf,rc->nkfc", Z, a, b, c)


def scale_to_energy(Y, target_prop, n):
    """
    Scale a component so that its squared Frobenius energy matches the
    target modality-level energy proportion.

    The target is not multiplied by the number of entries. This keeps the
    total energy comparable across modalities.
    """
    if target_prop <= 0 or Y.size == 0:
        return np.zeros_like(Y), 0.0

    current = float(np.sum(Y**2))
    target = float(n * target_prop)

    if current <= 1e-12:
        return Y.copy(), 1.0

    scale = np.sqrt(target / current)
    return Y * scale, float(scale)


def orthogonalize_loading_array(specific, shared, target_norm=1.0):
    """
    Orthogonalize modality-specific loadings against the shared loading span.

    This is used for sMRI arrays of shape (L, K, q) and fMRI arrays of
    shape (L, K, K).
    """
    if specific.size == 0 or shared.size == 0:
        return specific

    out = np.zeros_like(specific)
    flat_shared = shared.reshape(shared.shape[0], -1).T
    Q, _ = np.linalg.qr(flat_shared)

    for ell in range(specific.shape[0]):
        v = specific[ell].reshape(-1)
        v = v - Q @ (Q.T @ v)

        if np.linalg.norm(v) < 1e-12:
            v = specific[ell].reshape(-1)

        out[ell] = normalize(v, target_norm).reshape(specific.shape[1:])

    return out


def cp_tensor(a, b, c):
    return np.einsum("k,f,c->kfc", a, b, c)


def rank1_cp_als(T, n_iter=30):
    """
    Approximate a tensor by a rank-1 CP factorization.

    The amplitude is absorbed into the spatial factor.
    """
    K, F, C = T.shape

    U, _, _ = np.linalg.svd(T.reshape(K, F * C), full_matrices=False)

    a = U[:, 0]
    b = np.ones(F) / np.sqrt(F)
    c = np.ones(C) / np.sqrt(C)

    for _ in range(n_iter):
        a = np.einsum("kfc,f,c->k", T, b, c)
        a = normalize(a)

        b = np.einsum("kfc,k,c->f", T, a, c)
        b = normalize(b)

        c = np.einsum("kfc,k,f->c", T, a, b)
        c = normalize(c)

    amp = float(np.einsum("kfc,k,f,c->", T, a, b, c))

    if amp < 0:
        amp = -amp
        a = -a

    return amp * a, b, c


def orthogonalize_eeg_specific_cp(a_specific, b_specific, c_specific,
                                  a_shared, b_shared, c_shared):
    """
    Orthogonalize EEG-specific CP loading tensors against the shared EEG
    loading span, then project each tensor back to rank one.
    """
    if a_specific.size == 0 or a_shared.size == 0:
        return a_specific, b_specific, c_specific

    shared_vecs = [
        cp_tensor(a_shared[r], b_shared[r], c_shared[r]).reshape(-1)
        for r in range(a_shared.shape[0])
    ]

    Q, _ = np.linalg.qr(np.column_stack(shared_vecs))

    a_out = np.zeros_like(a_specific)
    b_out = np.zeros_like(b_specific)
    c_out = np.zeros_like(c_specific)

    for ell in range(a_specific.shape[0]):
        T = cp_tensor(a_specific[ell], b_specific[ell], c_specific[ell])
        v = T.reshape(-1)
        v = v - Q @ (Q.T @ v)

        a, b, c = rank1_cp_als(v.reshape(T.shape), n_iter=30)

        a_out[ell] = normalize(a)
        b_out[ell] = normalize(b)
        c_out[ell] = normalize(c)

    return a_out, b_out, c_out


def simulate_age(Z, rng, mean, sd, noise_sd):
    n, R = Z.shape

    beta = rng.normal(size=R)
    beta = beta / max(np.linalg.norm(beta), 1e-12)

    age = Z @ beta + noise_sd * rng.normal(size=n)
    age = (age - age.mean()) / max(age.std(), 1e-12)

    return mean + sd * age


def add_noise(n, K, q, F, C, rng):
    S_noise = rng.normal(size=(n, K, q))
    F_noise = np.array([symmetrize(rng.normal(size=(K, K))) for _ in range(n)])
    E_noise = rng.normal(size=(n, K, F, C))

    return S_noise, F_noise, E_noise


def make_subject_info(n, F, C):
    subject_ids = np.array([f"sub-{i + 1:04d}" for i in range(n)])

    smri_measures = np.array(
        ["thickness", "area", "sulcal_depth", "mean_curvature"],
        dtype=object,
    )

    eeg_freq_centers = np.linspace(2.0, 30.0, F)
    eeg_freq_bands = np.array([f"f{j + 1}" for j in range(F)], dtype=object)
    condition_names = np.array(["EO", "EC"][:C], dtype=object)

    return subject_ids, smri_measures, eeg_freq_centers, eeg_freq_bands, condition_names


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = args.n
    K = args.K
    q = args.q
    F = args.F
    C = args.C
    R = args.R

    prop_shared, prop_specific, prop_noise = SCENARIOS[args.scenario]["energy"]

    default_Ls, default_Lf, default_Le = SCENARIOS[args.scenario]["specific_ranks"]
    Ls = default_Ls if args.Ls is None else args.Ls
    Lf = default_Lf if args.Lf is None else args.Lf
    Le = default_Le if args.Le is None else args.Le

    lambda_vals = np.arange(1, K + 1, dtype=float)
    decay = lambda_vals ** (-0.5 * args.lambda_power)

    # Scores
    Z = sample_scores(n, R, rng)
    U_s = sample_scores(n, Ls, rng, orthogonal_to=Z)
    U_f = sample_scores(n, Lf, rng, orthogonal_to=Z)
    U_e = sample_scores(n, Le, rng, orthogonal_to=Z)

    # Loadings
    A_shared = sample_smri_loadings(R, K, q, decay, rng)
    M_shared = sample_fmri_loadings(R, K, decay, rng)
    a_shared, b_shared, c_shared = sample_eeg_loadings(R, K, F, C, decay, rng)

    A_specific = sample_smri_loadings(Ls, K, q, decay, rng)
    M_specific = sample_fmri_loadings(Lf, K, decay, rng)
    a_specific, b_specific, c_specific = sample_eeg_loadings(Le, K, F, C, decay, rng)

    # Make the modality-specific loadings approximately distinct from the shared loadings.
    A_specific = orthogonalize_loading_array(A_specific, A_shared)
    M_specific = orthogonalize_loading_array(M_specific, M_shared)
    a_specific, b_specific, c_specific = orthogonalize_eeg_specific_cp(
        a_specific,
        b_specific,
        c_specific,
        a_shared,
        b_shared,
        c_shared,
    )

    # Components before scaling
    S_shared_raw = reconstruct_smri(Z, A_shared)
    F_shared_raw = reconstruct_fmri(Z, M_shared)
    E_shared_raw = reconstruct_eeg(Z, a_shared, b_shared, c_shared)

    S_specific_raw = reconstruct_smri(U_s, A_specific)
    F_specific_raw = reconstruct_fmri(U_f, M_specific)
    E_specific_raw = reconstruct_eeg(U_e, a_specific, b_specific, c_specific)

    S_noise_raw, F_noise_raw, E_noise_raw = add_noise(n, K, q, F, C, rng)

    # Scale components to the requested energy proportions.
    S_shared, sc_S_shared = scale_to_energy(S_shared_raw, prop_shared, n)
    S_specific, sc_S_specific = scale_to_energy(S_specific_raw, prop_specific, n)
    S_noise, _ = scale_to_energy(S_noise_raw, prop_noise, n)

    F_shared, sc_F_shared = scale_to_energy(F_shared_raw, prop_shared, n)
    F_specific, sc_F_specific = scale_to_energy(F_specific_raw, prop_specific, n)
    F_noise, _ = scale_to_energy(F_noise_raw, prop_noise, n)

    E_shared, sc_E_shared = scale_to_energy(E_shared_raw, prop_shared, n)
    E_specific, sc_E_specific = scale_to_energy(E_specific_raw, prop_specific, n)
    E_noise, _ = scale_to_energy(E_noise_raw, prop_noise, n)

    # Keep the saved true loadings on the same scale as the simulated data.
    A_shared *= sc_S_shared
    A_specific *= sc_S_specific
    M_shared *= sc_F_shared
    M_specific *= sc_F_specific
    a_shared *= sc_E_shared
    a_specific *= sc_E_specific

    S_tilde = S_shared + S_specific + S_noise
    Sigma_tilde = F_shared + F_specific + F_noise
    E = E_shared + E_specific + E_noise

    # Natural-scale versions, included for completeness.
    tau = rng.uniform(0.7, 1.5, size=q)
    S = S_tilde * tau[None, None, :]
    Sigma = Sigma_tilde.copy()
    P = np.exp(np.clip(E, -20, 20))

    age = simulate_age(
        Z,
        rng,
        mean=args.age_mean,
        sd=args.age_sd,
        noise_sd=args.age_noise_sd,
    )
    male = rng.binomial(1, 0.5, size=n).astype(float)

    (
        subject_ids,
        smri_measures,
        eeg_freq_centers,
        eeg_freq_bands,
        condition_names,
    ) = make_subject_info(n, F, C)

    np.savez_compressed(
        outdir / "simulated_multiview_data.npz",
        subject_ids=subject_ids,
        S_tilde=S_tilde,
        Sigma_tilde=Sigma_tilde,
        E=E,
        S=S,
        Sigma=Sigma,
        P=P,
        n=np.array(n),
        K=np.array(K),
        q=np.array(q),
        F=np.array(F),
        C=np.array(C),
        lambda_vals=lambda_vals,
        tau=tau,
        smri_measures=smri_measures,
        eeg_freq_centers=eeg_freq_centers,
        eeg_freq_bands=eeg_freq_bands,
        condition_names=condition_names,
        age=age,
        male=male,
        Z_true=Z,
        U_s_true=U_s,
        U_f_true=U_f,
        U_e_true=U_e,
        A_shared_true=A_shared,
        M_shared_true=M_shared,
        a_shared_true=a_shared,
        b_shared_true=b_shared,
        c_shared_true=c_shared,
        A_specific_true=A_specific,
        M_specific_true=M_specific,
        a_specific_true=a_specific,
        b_specific_true=b_specific,
        c_specific_true=c_specific,
        scenario=np.array(args.scenario),
        prop_shared=np.array(prop_shared),
        prop_specific=np.array(prop_specific),
        prop_noise=np.array(prop_noise),
        true_R=np.array(R),
        true_Ls=np.array(Ls),
        true_Lf=np.array(Lf),
        true_Le=np.array(Le),
    )

    metadata = {
        "scenario": args.scenario,
        "seed": args.seed,
        "n": n,
        "K": K,
        "q": q,
        "F": F,
        "C": C,
        "R": R,
        "Ls": Ls,
        "Lf": Lf,
        "Le": Le,
        "energy_proportions": {
            "shared": prop_shared,
            "specific": prop_specific,
            "noise": prop_noise,
        },
        "notes": (
            "Components were generated on estimator-scale objects and rescaled "
            "to target squared-Frobenius energy proportions within each modality."
        ),
    }

    with open(outdir / "simulation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Wrote:", outdir / "simulated_multiview_data.npz")
    print("Wrote:", outdir / "simulation_metadata.json")
    print(f"scenario={args.scenario} n={n} props=({prop_shared}, {prop_specific}, {prop_noise})")
    print(f"shapes: S={S_tilde.shape}, Sigma={Sigma_tilde.shape}, E={E.shape}")


if __name__ == "__main__":
    main()