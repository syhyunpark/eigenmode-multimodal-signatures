"""checks for common and modality-specific cortical dimensions."""
from __future__ import annotations

import unittest

import numpy as np

from gemf import FixedRankGEMF, FixedRankGEMFConfig, MultiViewData
from gemf.fixed_rank_model import FixedRankGMHF, FixedRankGMHFConfig
from gemf.workflows import fit_gemf_model, infer_consistent_scores, subset_npz_dict


def make_data(ks: int, kf: int, ke: int) -> MultiViewData:
    rng = np.random.default_rng(73)
    n = 14
    covariance = rng.normal(size=(n, kf, kf))
    return MultiViewData(
        S_tilde=rng.normal(size=(n, ks, 4)),
        Sigma_tilde=(covariance + covariance.transpose(0, 2, 1)) / 2,
        E=rng.normal(size=(n, ke, 5, 2)),
        lambda_vals=np.arange(1, max(ks, kf, ke) + 1, dtype=float),
        subject_ids=np.array([f"synthetic_{i:02d}" for i in range(n)]),
    )


def fit_data(data: MultiViewData, specific_rank: int = 1):
    return fit_gemf_model(
        data, R=2, Ls=specific_rank, Lf=specific_rank, Le=specific_rank,
        w_s=1.0, w_f=1.0, w_e=1.0,
        lambda_s=0.01, lambda_f=0.01, lambda_e=0.01, zeta_e=0.01,
        specific_ridge=1e-6, score_ridge=0.1,
        max_iter=3, tol=1e-5, seed=0, quiet=True,
    )


class DimensionCompatibilityTest(unittest.TestCase):
    def test_mixed_dimensions_fit_reconstruction_and_scores(self):
        for dimensions in ((20, 20, 20), (50, 50, 20), (6, 9, 4)):
            with self.subTest(dimensions=dimensions):
                ks, kf, ke = dimensions
                data = make_data(*dimensions)
                fit, _, model = fit_data(data)
                self.assertEqual(fit["A_shared"].shape, (2, ks, 4))
                self.assertEqual(fit["M_shared"].shape, (2, kf, kf))
                self.assertEqual(fit["a_shared"].shape, (2, ke))
                np.testing.assert_array_equal(fit["subject_ids"], data.subject_ids)
                for reconstruction, shape in (
                    (model.reconstruct_total_smri(), data.S_tilde.shape),
                    (model.reconstruct_total_fmri(), data.Sigma_tilde.shape),
                    (model.reconstruct_total_eeg(), data.E.shape),
                ):
                    self.assertEqual(reconstruction.shape, shape)
                    self.assertTrue(np.isfinite(reconstruction).all())
                scores = infer_consistent_scores(
                    fit, data.subset(np.array([0, 3, 9])),
                    w_s=1.0, w_f=1.0, w_e=1.0, score_ridge=0.1,
                )
                for name, dimension in (("Zhat", 2), ("score", 5)):
                    self.assertEqual(scores[name].shape, (3, dimension))
                    self.assertTrue(np.isfinite(scores[name]).all())

    def test_no_specific_components_with_mixed_dimensions(self):
        data = make_data(6, 9, 4)
        fit, _, model = fit_data(data, specific_rank=0)
        for value, shape in (
            (model.reconstruct_specific_smri(), data.S_tilde.shape),
            (model.reconstruct_specific_fmri(), data.Sigma_tilde.shape),
            (model.reconstruct_specific_eeg(), data.E.shape),
        ):
            self.assertEqual(value.shape, shape)
            np.testing.assert_array_equal(value, np.zeros(shape))
        scores = infer_consistent_scores(
            fit, data, w_s=1.0, w_f=1.0, w_e=1.0, score_ridge=0.1,
        )
        np.testing.assert_allclose(scores["Zhat"], scores["score"], rtol=1e-6)

    def test_legacy_names_and_common_dimension_are_preserved(self):
        self.assertIs(FixedRankGMHF, FixedRankGEMF)
        self.assertIs(FixedRankGMHFConfig, FixedRankGEMFConfig)
        data = make_data(4, 4, 4)
        self.assertEqual(data.K, 4)
        _, _, model = fit_data(data)
        self.assertEqual(model.K, 4)
        with self.assertRaisesRegex(ValueError, "K_s, K_f, and K_e"):
            _ = make_data(6, 9, 4).K

    def test_subsetting_preserves_mode_weights_when_n_equals_mode_count(self):
        data = make_data(14, 14, 4)
        indices = np.array([1, 4, 7])
        result = subset_npz_dict(
            {"subject_ids": data.subject_ids, "S_tilde": data.S_tilde,
             "Sigma_tilde": data.Sigma_tilde, "E": data.E,
             "lambda_vals": data.lambda_vals},
            indices,
        )
        np.testing.assert_array_equal(result["lambda_vals"], data.lambda_vals)
        np.testing.assert_array_equal(result["subject_ids"], data.subject_ids[indices])
        self.assertEqual(result["S_tilde"].shape[0], len(indices))


if __name__ == "__main__":
    unittest.main()
