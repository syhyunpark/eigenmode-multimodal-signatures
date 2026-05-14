# gemf/fixed_rank_model.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np

from .data import MultiViewData


def _symmetrize(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + x.T)


def _safe_norm(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(max(np.linalg.norm(x), eps))


def _make_second_difference(F: int) -> np.ndarray:
    if F < 3:
        return np.zeros((0, F), dtype=float)
    D = np.zeros((F - 2, F), dtype=float)
    for i in range(F - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _standardize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Center columns and scale them to unit variance.
    Returns standardized matrix and the scales used.
    """
    if X.size == 0:
        return X.copy(), np.zeros(0, dtype=float)

    Y = X.copy()
    Y -= Y.mean(axis=0, keepdims=True)
    sds = Y.std(axis=0, ddof=0)
    sds = np.maximum(sds, 1e-8)
    Y /= sds[None, :]
    return Y, sds


def _leading_rank1_factors(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crude HOSVD-style rank-1 initialization for a K x F x C tensor.
    """
    K, F, C = tensor.shape

    U0, _, _ = np.linalg.svd(tensor.reshape(K, F * C), full_matrices=False)
    a = U0[:, 0]

    T_fc = np.tensordot(a, tensor, axes=(0, 0))  # (F, C)
    U1, _, Vt1 = np.linalg.svd(T_fc, full_matrices=False)
    b = U1[:, 0]
    c = Vt1.T[:, 0]

    b = b / _safe_norm(b)
    c = c / _safe_norm(c)
    a = a / _safe_norm(a)
    return a, b, c


@dataclass
class FixedRankGMHFConfig:
    R: int
    Ls: int = 0
    Lf: int = 0
    Le: int = 0

    w_s: float = 1.0
    w_f: float = 1.0
    w_e: float = 1.0

    lambda_s: float = 1e-2
    lambda_f: float = 1e-2
    lambda_e: float = 1e-2
    zeta_e: float = 1e-2

    specific_ridge: float = 1e-6
    score_ridge: float = 1e-8

    prefit_shared_iters: int = 0

    max_iter: int = 100
    tol: float = 1e-5
    random_state: int = 0
    verbose: bool = True


class FixedRankGMHF:
    """
    Full fixed-rank geometry-aligned multiview harmonic factorization.

    Shared part:
        S_tilde_i      ~= sum_r z_ir A_r^(s)
        Sigma_tilde_i  ~= sum_r z_ir M_r
        E_i            ~= sum_r z_ir (a_r ∘ b_r ∘ c_r)

    Modality-specific parts:
        S_tilde_i      += sum_l u_il^(s) A_l^(s,specific)
        Sigma_tilde_i  += sum_l u_il^(f) M_l^(specific)
        E_i            += sum_l u_il^(e) (a_l^(specific) ∘ b_l^(specific) ∘ c_l^(specific))
    """

    def __init__(self, config: FixedRankGMHFConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

        # learned parameters
        self.Z: Optional[np.ndarray] = None
        self.U_s: Optional[np.ndarray] = None
        self.U_f: Optional[np.ndarray] = None
        self.U_e: Optional[np.ndarray] = None

        self.A_shared: Optional[np.ndarray] = None          # (R, K, q)
        self.A_specific_s: Optional[np.ndarray] = None      # (Ls, K, q)

        self.M_shared: Optional[np.ndarray] = None          # (R, K, K)
        self.M_specific_f: Optional[np.ndarray] = None      # (Lf, K, K)

        self.a_shared: Optional[np.ndarray] = None          # (R, K)
        self.b_shared: Optional[np.ndarray] = None          # (R, F)
        self.c_shared: Optional[np.ndarray] = None          # (R, C)

        self.a_specific_e: Optional[np.ndarray] = None      # (Le, K)
        self.b_specific_e: Optional[np.ndarray] = None      # (Le, F)
        self.c_specific_e: Optional[np.ndarray] = None      # (Le, C)

        # dimensions
        self.n: Optional[int] = None
        self.K: Optional[int] = None
        self.q: Optional[int] = None
        self.F: Optional[int] = None
        self.C: Optional[int] = None
        self.lambda_vals: Optional[np.ndarray] = None
        self.D_f: Optional[np.ndarray] = None
        self.DfTDf: Optional[np.ndarray] = None

        self.history_: Dict[str, list] = {
            "loss_total": [],
            "loss_s": [],
            "loss_f": [],
            "loss_e": [],
            "penalty": [],
        }

    def _check_fitted(self) -> None:
        if self.Z is None:
            raise RuntimeError("Model is not fitted yet.")

    def _has_specific_components(self) -> bool:
        return (self.config.Ls > 0) or (self.config.Lf > 0) or (self.config.Le > 0)

    # ------------------------------------------------------------------
    # Reconstruction helpers
    # ------------------------------------------------------------------
    def reconstruct_shared_smri(self) -> np.ndarray:
        if self.config.R == 0:
            return np.zeros((self.n, self.K, self.q), dtype=float)
        return np.einsum("nr,rkq->nkq", self.Z, self.A_shared)

    def reconstruct_specific_smri(self) -> np.ndarray:
        if self.config.Ls == 0:
            return np.zeros((self.n, self.K, self.q), dtype=float)
        return np.einsum("nl,lkq->nkq", self.U_s, self.A_specific_s)

    def reconstruct_total_smri(self) -> np.ndarray:
        return self.reconstruct_shared_smri() + self.reconstruct_specific_smri()

    def reconstruct_shared_fmri(self) -> np.ndarray:
        if self.config.R == 0:
            return np.zeros((self.n, self.K, self.K), dtype=float)
        return np.einsum("nr,rkm->nkm", self.Z, self.M_shared)

    def reconstruct_specific_fmri(self) -> np.ndarray:
        if self.config.Lf == 0:
            return np.zeros((self.n, self.K, self.K), dtype=float)
        return np.einsum("nl,lkm->nkm", self.U_f, self.M_specific_f)

    def reconstruct_total_fmri(self) -> np.ndarray:
        return self.reconstruct_shared_fmri() + self.reconstruct_specific_fmri()

    def _shared_factor_tensor(self, r: int) -> np.ndarray:
        return np.einsum("k,f,c->kfc", self.a_shared[r], self.b_shared[r], self.c_shared[r])

    def _specific_factor_tensor(self, ell: int) -> np.ndarray:
        return np.einsum(
            "k,f,c->kfc",
            self.a_specific_e[ell],
            self.b_specific_e[ell],
            self.c_specific_e[ell],
        )

    def reconstruct_shared_eeg(self) -> np.ndarray:
        if self.config.R == 0:
            return np.zeros((self.n, self.K, self.F, self.C), dtype=float)
        return np.einsum("nr,rk,rf,rc->nkfc", self.Z, self.a_shared, self.b_shared, self.c_shared)

    def reconstruct_specific_eeg(self) -> np.ndarray:
        if self.config.Le == 0:
            return np.zeros((self.n, self.K, self.F, self.C), dtype=float)
        return np.einsum(
            "nl,lk,lf,lc->nkfc",
            self.U_e, self.a_specific_e, self.b_specific_e, self.c_specific_e
        )

    def reconstruct_total_eeg(self) -> np.ndarray:
        return self.reconstruct_shared_eeg() + self.reconstruct_specific_eeg()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, data: MultiViewData) -> None:
        self.n, self.K, self.q = data.S_tilde.shape
        _, _, self.F, self.C = data.E.shape
        self.lambda_vals = data.lambda_vals.copy()
        self.D_f = _make_second_difference(self.F)
        self.DfTDf = self.D_f.T @ self.D_f if self.D_f.size > 0 else np.zeros((self.F, self.F))

        R, Ls, Lf, Le = self.config.R, self.config.Ls, self.config.Lf, self.config.Le

        # -------- initialize shared scores from concatenated SVD --------
        X_s = data.S_tilde.reshape(self.n, -1)
        X_f = data.Sigma_tilde.reshape(self.n, -1)
        X_e = data.E.reshape(self.n, -1)

        X_concat = np.concatenate(
            [
                np.sqrt(self.config.w_s) * X_s,
                np.sqrt(self.config.w_f) * X_f,
                np.sqrt(self.config.w_e) * X_e,
            ],
            axis=1,
        )
        X_concat = X_concat - X_concat.mean(axis=0, keepdims=True)

        U, _, _ = np.linalg.svd(X_concat, full_matrices=False)
        Z0 = np.sqrt(self.n) * U[:, :R]
        Z0, _ = _standardize_columns(Z0)
        self.Z = Z0

        # -------- initialize shared loadings --------
        self.A_shared = np.zeros((R, self.K, self.q), dtype=float)
        self.M_shared = np.zeros((R, self.K, self.K), dtype=float)
        self.a_shared = np.zeros((R, self.K), dtype=float)
        self.b_shared = np.zeros((R, self.F), dtype=float)
        self.c_shared = np.zeros((R, self.C), dtype=float)

        # sMRI/fMRI shared loadings by projection
        for r in range(R):
            zr = self.Z[:, r]
            num_s = np.sum(zr[:, None, None] * data.S_tilde, axis=0)
            denom_s = zr.dot(zr) + self.config.lambda_s * self.lambda_vals
            self.A_shared[r] = num_s / denom_s[:, None]

            num_f = np.sum(zr[:, None, None] * data.Sigma_tilde, axis=0)
            denom_f = zr.dot(zr) + self.config.lambda_f * (
                self.lambda_vals[:, None] + self.lambda_vals[None, :]
            )
            self.M_shared[r] = _symmetrize(num_f / denom_f)

        # EEG shared loadings by weighted rank-1 initialization
        decay = 1.0 / np.sqrt(np.maximum(self.lambda_vals, 1.0))
        for r in range(R):
            weighted_avg = np.einsum("n,nkfc->kfc", self.Z[:, r], data.E)
            a, b, c = _leading_rank1_factors(weighted_avg)
            a = decay * a
            a = a / _safe_norm(a)
            self.a_shared[r] = a
            self.b_shared[r] = b / _safe_norm(b)
            self.c_shared[r] = c / _safe_norm(c)

        # -------- initialize modality-specific parts as zeros --------
        # These will be initialized from shared residuals either immediately
        # or after a short shared-only prefit stage.
        self.U_s = np.zeros((self.n, Ls), dtype=float)
        self.U_f = np.zeros((self.n, Lf), dtype=float)
        self.U_e = np.zeros((self.n, Le), dtype=float)

        self.A_specific_s = np.zeros((Ls, self.K, self.q), dtype=float)
        self.M_specific_f = np.zeros((Lf, self.K, self.K), dtype=float)
        self.a_specific_e = np.zeros((Le, self.K), dtype=float)
        self.b_specific_e = np.zeros((Le, self.F), dtype=float)
        self.c_specific_e = np.zeros((Le, self.C), dtype=float)

    def _initialize_specific_from_shared_residuals(self, data: MultiViewData) -> None:
        """
        Initialize modality-specific components from residuals after removing
        the current shared fit.
        """
        Ls, Lf, Le = self.config.Ls, self.config.Lf, self.config.Le

        # reset current specific components
        if Ls > 0:
            self.U_s = np.zeros((self.n, Ls), dtype=float)
            self.A_specific_s = np.zeros((Ls, self.K, self.q), dtype=float)
        if Lf > 0:
            self.U_f = np.zeros((self.n, Lf), dtype=float)
            self.M_specific_f = np.zeros((Lf, self.K, self.K), dtype=float)
        if Le > 0:
            self.U_e = np.zeros((self.n, Le), dtype=float)
            self.a_specific_e = np.zeros((Le, self.K), dtype=float)
            self.b_specific_e = np.zeros((Le, self.F), dtype=float)
            self.c_specific_e = np.zeros((Le, self.C), dtype=float)

        # sMRI-specific init
        if Ls > 0:
            R_s = data.S_tilde - self.reconstruct_shared_smri()
            X = R_s.reshape(self.n, -1)
            X = X - X.mean(axis=0, keepdims=True)
            Umat, _, _ = np.linalg.svd(X, full_matrices=False)
            Us0 = np.sqrt(self.n) * Umat[:, :Ls]
            self.U_s, _ = _standardize_columns(Us0)

            for ell in range(Ls):
                u = self.U_s[:, ell]
                num = np.sum(u[:, None, None] * R_s, axis=0)
                denom = u.dot(u) + self.config.specific_ridge
                self.A_specific_s[ell] = num / denom

        # fMRI-specific init
        if Lf > 0:
            R_f = data.Sigma_tilde - self.reconstruct_shared_fmri()
            X = R_f.reshape(self.n, -1)
            X = X - X.mean(axis=0, keepdims=True)
            Umat, _, _ = np.linalg.svd(X, full_matrices=False)
            Uf0 = np.sqrt(self.n) * Umat[:, :Lf]
            self.U_f, _ = _standardize_columns(Uf0)

            for ell in range(Lf):
                u = self.U_f[:, ell]
                num = np.sum(u[:, None, None] * R_f, axis=0)
                denom = u.dot(u) + self.config.specific_ridge
                self.M_specific_f[ell] = _symmetrize(num / denom)

        # EEG-specific init
        if Le > 0:
            R_e = data.E - self.reconstruct_shared_eeg()
            X = R_e.reshape(self.n, -1)
            X = X - X.mean(axis=0, keepdims=True)
            Umat, _, _ = np.linalg.svd(X, full_matrices=False)
            Ue0 = np.sqrt(self.n) * Umat[:, :Le]
            self.U_e, _ = _standardize_columns(Ue0)

            for ell in range(Le):
                weighted_avg = np.einsum("n,nkfc->kfc", self.U_e[:, ell], R_e)
                a, b, c = _leading_rank1_factors(weighted_avg)
                self.a_specific_e[ell] = a / _safe_norm(a)
                self.b_specific_e[ell] = b / _safe_norm(b)
                self.c_specific_e[ell] = c / _safe_norm(c)

    # ------------------------------------------------------------------
    # Score updates
    # ------------------------------------------------------------------
    def _update_shared_scores(self, data: MultiViewData) -> None:
        R = self.config.R
        if R == 0:
            return

        S_res = data.S_tilde - self.reconstruct_specific_smri()
        F_res = data.Sigma_tilde - self.reconstruct_specific_fmri()
        E_res = data.E - self.reconstruct_specific_eeg()

        A_flat = self.A_shared.reshape(R, -1)
        M_flat = self.M_shared.reshape(R, -1)
        G_flat = np.array([self._shared_factor_tensor(r).reshape(-1) for r in range(R)])

        Gram = (
            self.config.w_s * (A_flat @ A_flat.T)
            + self.config.w_f * (M_flat @ M_flat.T)
            + self.config.w_e * (G_flat @ G_flat.T)
        )
        Gram += self.config.score_ridge * np.eye(R)

        rhs_s = self.config.w_s * np.einsum("nkq,rkq->nr", S_res, self.A_shared)
        rhs_f = self.config.w_f * np.einsum("nkm,rkm->nr", F_res, self.M_shared)
        rhs_e = self.config.w_e * np.einsum(
            "nkfc,rkfc->nr",
            E_res,
            np.array([self._shared_factor_tensor(r) for r in range(R)]),
        )
        RHS = rhs_s + rhs_f + rhs_e

        Z_new = np.linalg.solve(Gram, RHS.T).T

        # standardize columns and absorb scale into shared loadings
        Z_new, sds = _standardize_columns(Z_new)
        for r in range(R):
            self.A_shared[r] *= sds[r]
            self.M_shared[r] *= sds[r]
            self.a_shared[r] *= sds[r]
        self.Z = Z_new

    def _update_specific_scores_smri(self, data: MultiViewData) -> None:
        if self.config.Ls == 0:
            return

        S_res = data.S_tilde - self.reconstruct_shared_smri()
        Ls = self.config.Ls
        B_flat = self.A_specific_s.reshape(Ls, -1)

        Gram = B_flat @ B_flat.T + self.config.score_ridge * np.eye(Ls)
        RHS = np.einsum("nkq,lkq->nl", S_res, self.A_specific_s)

        U_new = np.linalg.solve(Gram, RHS.T).T
        U_new, sds = _standardize_columns(U_new)
        for ell in range(Ls):
            self.A_specific_s[ell] *= sds[ell]
        self.U_s = U_new

    def _update_specific_scores_fmri(self, data: MultiViewData) -> None:
        if self.config.Lf == 0:
            return

        F_res = data.Sigma_tilde - self.reconstruct_shared_fmri()
        Lf = self.config.Lf
        B_flat = self.M_specific_f.reshape(Lf, -1)

        Gram = B_flat @ B_flat.T + self.config.score_ridge * np.eye(Lf)
        RHS = np.einsum("nkm,lkm->nl", F_res, self.M_specific_f)

        U_new = np.linalg.solve(Gram, RHS.T).T
        U_new, sds = _standardize_columns(U_new)
        for ell in range(Lf):
            self.M_specific_f[ell] *= sds[ell]
        self.U_f = U_new

    def _update_specific_scores_eeg(self, data: MultiViewData) -> None:
        if self.config.Le == 0:
            return

        E_res = data.E - self.reconstruct_shared_eeg()
        Le = self.config.Le
        G_flat = np.array([self._specific_factor_tensor(ell).reshape(-1) for ell in range(Le)])

        Gram = G_flat @ G_flat.T + self.config.score_ridge * np.eye(Le)
        RHS = np.einsum(
            "nkfc,lkfc->nl",
            E_res,
            np.array([self._specific_factor_tensor(ell) for ell in range(Le)]),
        )

        U_new = np.linalg.solve(Gram, RHS.T).T
        U_new, sds = _standardize_columns(U_new)
        for ell in range(Le):
            self.a_specific_e[ell] *= sds[ell]
        self.U_e = U_new

    # ------------------------------------------------------------------
    # Loading updates
    # ------------------------------------------------------------------
    def _update_shared_smri_loadings(self, data: MultiViewData) -> None:
        if self.config.R == 0:
            return

        specific_recon = self.reconstruct_specific_smri()
        shared_recon = self.reconstruct_shared_smri()

        for r in range(self.config.R):
            zr = self.Z[:, r]
            T = data.S_tilde - specific_recon - shared_recon + zr[:, None, None] * self.A_shared[r][None, :, :]
            num = np.sum(zr[:, None, None] * T, axis=0)
            denom = zr.dot(zr) + self.config.lambda_s * self.lambda_vals
            A_new = num / denom[:, None]

            shared_recon = shared_recon - zr[:, None, None] * self.A_shared[r][None, :, :] + zr[:, None, None] * A_new[None, :, :]
            self.A_shared[r] = A_new

    def _update_specific_smri_loadings(self, data: MultiViewData) -> None:
        if self.config.Ls == 0:
            return

        shared_recon = self.reconstruct_shared_smri()
        specific_recon = self.reconstruct_specific_smri()

        for ell in range(self.config.Ls):
            u = self.U_s[:, ell]
            T = data.S_tilde - shared_recon - specific_recon + u[:, None, None] * self.A_specific_s[ell][None, :, :]
            num = np.sum(u[:, None, None] * T, axis=0)
            denom = u.dot(u) + self.config.specific_ridge
            A_new = num / denom

            specific_recon = specific_recon - u[:, None, None] * self.A_specific_s[ell][None, :, :] + u[:, None, None] * A_new[None, :, :]
            self.A_specific_s[ell] = A_new

    def _update_shared_fmri_loadings(self, data: MultiViewData) -> None:
        if self.config.R == 0:
            return

        specific_recon = self.reconstruct_specific_fmri()
        shared_recon = self.reconstruct_shared_fmri()
        lam_sum = self.lambda_vals[:, None] + self.lambda_vals[None, :]

        for r in range(self.config.R):
            zr = self.Z[:, r]
            T = data.Sigma_tilde - specific_recon - shared_recon + zr[:, None, None] * self.M_shared[r][None, :, :]
            num = np.sum(zr[:, None, None] * T, axis=0)
            denom = zr.dot(zr) + self.config.lambda_f * lam_sum
            M_new = _symmetrize(num / denom)

            shared_recon = shared_recon - zr[:, None, None] * self.M_shared[r][None, :, :] + zr[:, None, None] * M_new[None, :, :]
            self.M_shared[r] = M_new

    def _update_specific_fmri_loadings(self, data: MultiViewData) -> None:
        if self.config.Lf == 0:
            return

        shared_recon = self.reconstruct_shared_fmri()
        specific_recon = self.reconstruct_specific_fmri()

        for ell in range(self.config.Lf):
            u = self.U_f[:, ell]
            T = data.Sigma_tilde - shared_recon - specific_recon + u[:, None, None] * self.M_specific_f[ell][None, :, :]
            num = np.sum(u[:, None, None] * T, axis=0)
            denom = u.dot(u) + self.config.specific_ridge
            M_new = _symmetrize(num / denom)

            specific_recon = specific_recon - u[:, None, None] * self.M_specific_f[ell][None, :, :] + u[:, None, None] * M_new[None, :, :]
            self.M_specific_f[ell] = M_new

    def _update_one_shared_eeg_factor(self, data: MultiViewData, r: int, shared_recon: np.ndarray) -> np.ndarray:
        zr = self.Z[:, r]
        factor_old = self._shared_factor_tensor(r)
        specific_recon = self.reconstruct_specific_eeg()
        T = data.E - specific_recon - shared_recon + zr[:, None, None, None] * factor_old[None, :, :, :]

        a = self.a_shared[r].copy()
        b = self.b_shared[r].copy()
        c = self.c_shared[r].copy()

        proj_a = np.einsum("nkfc,f,c->nk", T, b, c)
        num_a = np.einsum("n,nk->k", zr, proj_a)
        denom_a = zr.dot(zr) * (b @ b) * (c @ c) + self.config.lambda_e * self.lambda_vals
        a = num_a / np.maximum(denom_a, 1e-10)

        proj_b = np.einsum("nkfc,k,c->nf", T, a, c)
        num_b = np.einsum("n,nf->f", zr, proj_b)
        coef_b = zr.dot(zr) * (a @ a) * (c @ c)
        mat_b = coef_b * np.eye(self.F) + self.config.zeta_e * self.DfTDf
        b = np.linalg.solve(mat_b + 1e-8 * np.eye(self.F), num_b)
        norm_b = _safe_norm(b)
        b = b / norm_b
        a = a * norm_b

        proj_c = np.einsum("nkfc,k,f->nc", T, a, b)
        num_c = np.einsum("n,nc->c", zr, proj_c)
        denom_c = zr.dot(zr) * (a @ a) * (b @ b)
        c = num_c / max(denom_c, 1e-10)
        norm_c = _safe_norm(c)
        c = c / norm_c
        a = a * norm_c

        self.a_shared[r] = a
        self.b_shared[r] = b
        self.c_shared[r] = c

        factor_new = self._shared_factor_tensor(r)
        shared_recon = shared_recon - zr[:, None, None, None] * factor_old[None, :, :, :] + zr[:, None, None, None] * factor_new[None, :, :, :]
        return shared_recon

    def _update_one_specific_eeg_factor(self, data: MultiViewData, ell: int, specific_recon: np.ndarray) -> np.ndarray:
        u = self.U_e[:, ell]
        factor_old = self._specific_factor_tensor(ell)
        shared_recon = self.reconstruct_shared_eeg()
        T = data.E - shared_recon - specific_recon + u[:, None, None, None] * factor_old[None, :, :, :]

        a = self.a_specific_e[ell].copy()
        b = self.b_specific_e[ell].copy()
        c = self.c_specific_e[ell].copy()

        proj_a = np.einsum("nkfc,f,c->nk", T, b, c)
        num_a = np.einsum("n,nk->k", u, proj_a)
        denom_a = u.dot(u) * (b @ b) * (c @ c) + self.config.specific_ridge
        a = num_a / np.maximum(denom_a, 1e-10)

        proj_b = np.einsum("nkfc,k,c->nf", T, a, c)
        num_b = np.einsum("n,nf->f", u, proj_b)
        coef_b = u.dot(u) * (a @ a) * (c @ c)
        b = num_b / max(coef_b + self.config.specific_ridge, 1e-10)
        norm_b = _safe_norm(b)
        b = b / norm_b
        a = a * norm_b

        proj_c = np.einsum("nkfc,k,f->nc", T, a, b)
        num_c = np.einsum("n,nc->c", u, proj_c)
        denom_c = u.dot(u) * (a @ a) * (b @ b)
        c = num_c / max(denom_c + self.config.specific_ridge, 1e-10)
        norm_c = _safe_norm(c)
        c = c / norm_c
        a = a * norm_c

        self.a_specific_e[ell] = a
        self.b_specific_e[ell] = b
        self.c_specific_e[ell] = c

        factor_new = self._specific_factor_tensor(ell)
        specific_recon = specific_recon - u[:, None, None, None] * factor_old[None, :, :, :] + u[:, None, None, None] * factor_new[None, :, :, :]
        return specific_recon

    def _update_shared_eeg_loadings(self, data: MultiViewData) -> None:
        if self.config.R == 0:
            return
        shared_recon = self.reconstruct_shared_eeg()
        for r in range(self.config.R):
            shared_recon = self._update_one_shared_eeg_factor(data, r, shared_recon)

    def _update_specific_eeg_loadings(self, data: MultiViewData) -> None:
        if self.config.Le == 0:
            return
        specific_recon = self.reconstruct_specific_eeg()
        for ell in range(self.config.Le):
            specific_recon = self._update_one_specific_eeg_factor(data, ell, specific_recon)

    # ------------------------------------------------------------------
    # Loss / penalties
    # ------------------------------------------------------------------
    def _compute_penalty(self) -> float:
        p_s = self.config.lambda_s * sum(
            np.trace(A.T @ np.diag(self.lambda_vals) @ A) for A in self.A_shared
        ) if self.config.R > 0 else 0.0

        p_f = self.config.lambda_f * sum(
            np.linalg.norm(np.sqrt(self.lambda_vals)[:, None] * M, ord="fro") ** 2
            + np.linalg.norm(M * np.sqrt(self.lambda_vals)[None, :], ord="fro") ** 2
            for M in self.M_shared
        ) if self.config.R > 0 else 0.0

        p_e = self.config.lambda_e * sum(
            a.T @ np.diag(self.lambda_vals) @ a for a in self.a_shared
        ) if self.config.R > 0 else 0.0

        p_b = self.config.zeta_e * sum(
            np.linalg.norm(self.D_f @ b) ** 2 for b in self.b_shared
        ) if (self.config.R > 0 and self.D_f.size > 0) else 0.0

        return float(p_s + p_f + p_e + p_b)

    def compute_loss(self, data: MultiViewData) -> Dict[str, float]:
        recon_s = self.reconstruct_total_smri()
        recon_f = self.reconstruct_total_fmri()
        recon_e = self.reconstruct_total_eeg()

        loss_s = self.config.w_s * float(np.sum((data.S_tilde - recon_s) ** 2))
        loss_f = self.config.w_f * float(np.sum((data.Sigma_tilde - recon_f) ** 2))
        loss_e = self.config.w_e * float(np.sum((data.E - recon_e) ** 2))
        penalty = self._compute_penalty()

        return {
            "loss_s": loss_s,
            "loss_f": loss_f,
            "loss_e": loss_e,
            "penalty": penalty,
            "loss_total": loss_s + loss_f + loss_e + penalty,
        }

    # ------------------------------------------------------------------
    # Shared-only prefit
    # ------------------------------------------------------------------
    def _run_shared_only_prefit(self, data: MultiViewData, n_iters: int) -> None:
        """
        Run a few shared-only iterations before initializing the modality-specific
        components. This often produces cleaner residuals for specific-factor
        initialization.
        """
        if n_iters <= 0 or self.config.R == 0:
            return

        if self.config.verbose:
            print(f"Running shared-only prefit for {n_iters} iterations...")

        prev_loss = np.inf
        for it in range(n_iters):
            self._update_shared_scores(data)
            self._update_shared_smri_loadings(data)
            self._update_shared_fmri_loadings(data)
            self._update_shared_eeg_loadings(data)

            losses = self.compute_loss(data)
            rel_change = abs(prev_loss - losses["loss_total"]) / max(abs(prev_loss), 1.0)
            prev_loss = losses["loss_total"]

            if self.config.verbose:
                print(
                    f"[prefit {it+1:03d}] "
                    f"total={losses['loss_total']:.6f} "
                    f"s={losses['loss_s']:.6f} "
                    f"f={losses['loss_f']:.6f} "
                    f"e={losses['loss_e']:.6f} "
                    f"pen={losses['penalty']:.6f} "
                    f"rel_change={rel_change:.3e}"
                )

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------
    def fit(self, data: MultiViewData) -> "FixedRankGMHF":
        if self.Z is None:
            self.initialize(data)

        # Two-stage strategy:
        # (1) optionally refine shared part alone
        # (2) initialize specific parts from shared residuals
        # (3) jointly refine full model
        if self._has_specific_components():
            if self.config.prefit_shared_iters > 0:
                self._run_shared_only_prefit(data, self.config.prefit_shared_iters)
            self._initialize_specific_from_shared_residuals(data)

        prev_loss = np.inf

        for it in range(self.config.max_iter):
            self._update_shared_scores(data)
            self._update_specific_scores_smri(data)
            self._update_specific_scores_fmri(data)
            self._update_specific_scores_eeg(data)

            self._update_shared_smri_loadings(data)
            self._update_specific_smri_loadings(data)

            self._update_shared_fmri_loadings(data)
            self._update_specific_fmri_loadings(data)

            self._update_shared_eeg_loadings(data)
            self._update_specific_eeg_loadings(data)

            losses = self.compute_loss(data)
            for k, v in losses.items():
                self.history_[k].append(v)

            rel_change = abs(prev_loss - losses["loss_total"]) / max(abs(prev_loss), 1.0)
            prev_loss = losses["loss_total"]

            if self.config.verbose:
                print(
                    f"[iter {it+1:03d}] "
                    f"total={losses['loss_total']:.6f} "
                    f"s={losses['loss_s']:.6f} "
                    f"f={losses['loss_f']:.6f} "
                    f"e={losses['loss_e']:.6f} "
                    f"pen={losses['penalty']:.6f} "
                    f"rel_change={rel_change:.3e}"
                )

            if rel_change < self.config.tol:
                if self.config.verbose:
                    print(f"Converged at iteration {it+1}.")
                break

        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def compute_mode_energy_curves(self, data: MultiViewData) -> Dict[str, np.ndarray]:
        B_s = np.mean(np.sum(data.S_tilde ** 2, axis=2), axis=0)
        B_f = np.mean(np.diagonal(data.Sigma_tilde, axis1=1, axis2=2), axis=0)
        B_e = np.mean(np.sum(np.exp(data.E), axis=(2, 3)), axis=0)

        W_s = np.cumsum(B_s) / max(np.sum(B_s), 1e-12)
        W_f = np.cumsum(B_f) / max(np.sum(B_f), 1e-12)
        W_e = np.cumsum(B_e) / max(np.sum(B_e), 1e-12)

        return {
            "B_s": B_s,
            "B_f": B_f,
            "B_e": B_e,
            "W_s": W_s,
            "W_f": W_f,
            "W_e": W_e,
        }

    def compute_sharedness_curves(self, data: MultiViewData) -> Dict[str, np.ndarray]:
        """
        Residual-based mode-wise sharedness after removing only the shared component.
        This makes the summaries less sensitive to the chosen modality-specific ranks.
        """
        shared_s = self.reconstruct_shared_smri()
        shared_f = self.reconstruct_shared_fmri()
        shared_e = self.reconstruct_shared_eeg()

        res_s = data.S_tilde - shared_s
        res_f = data.Sigma_tilde - shared_f
        res_e = data.E - shared_e

        num_s = np.sum(res_s ** 2, axis=(0, 2))
        den_s = np.sum(data.S_tilde ** 2, axis=(0, 2))
        SF_s = 1.0 - num_s / np.maximum(den_s, 1e-12)

        num_f = np.sum(res_f ** 2, axis=(0, 2))
        den_f = np.sum(data.Sigma_tilde ** 2, axis=(0, 2))
        SF_f = 1.0 - num_f / np.maximum(den_f, 1e-12)

        num_e = np.sum(res_e ** 2, axis=(0, 2, 3))
        den_e = np.sum(data.E ** 2, axis=(0, 2, 3))
        SF_e = 1.0 - num_e / np.maximum(den_e, 1e-12)

        return {
            "SF_s": SF_s,
            "SF_f": SF_f,
            "SF_e": SF_e,
        }

    def save_dict(self, data: Optional[MultiViewData] = None) -> Dict[str, Any]:
        self._check_fitted()
        out = {
            "config": asdict(self.config),
            "Z": self.Z,
            "U_s": self.U_s,
            "U_f": self.U_f,
            "U_e": self.U_e,
            "A_shared": self.A_shared,
            "A_specific_s": self.A_specific_s,
            "M_shared": self.M_shared,
            "M_specific_f": self.M_specific_f,
            "a_shared": self.a_shared,
            "b_shared": self.b_shared,
            "c_shared": self.c_shared,
            "a_specific_e": self.a_specific_e,
            "b_specific_e": self.b_specific_e,
            "c_specific_e": self.c_specific_e,
            "history_loss_total": np.asarray(self.history_["loss_total"]),
            "history_loss_s": np.asarray(self.history_["loss_s"]),
            "history_loss_f": np.asarray(self.history_["loss_f"]),
            "history_loss_e": np.asarray(self.history_["loss_e"]),
            "history_penalty": np.asarray(self.history_["penalty"]),
            "lambda_vals": self.lambda_vals,
        }
        if data is not None:
            out.update(self.compute_mode_energy_curves(data))
            out.update(self.compute_sharedness_curves(data))
        return out