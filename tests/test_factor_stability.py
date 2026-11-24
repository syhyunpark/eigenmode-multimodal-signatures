"""checks for the score/loading comparison and saved subsets."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from gemf.fixed_rank_model import FixedRankGEMFConfig
from gemf.score_inference import infer_shared_scores
from scripts.run_gemf_factor_stability import fit_once, make_data, read_splits
from scripts.summarize_gemf_factor_stability import (
    LOADINGS, METRICS, compare_factors, compare_fit, eeg_loading,
    orient_full_cohort, participant_summary,
)


class FactorComparisonTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.reference = dict(
            subject_ids=np.array([f"p{i}" for i in range(12)]),
            shared_scores=rng.normal(size=(12, 3)),
            A_shared=rng.normal(size=(3, 2, 4)),
            M_shared=rng.normal(size=(3, 2, 2)),
            a_shared=rng.normal(size=(3, 2)),
            b_shared=rng.normal(size=(3, 4)),
            c_shared=np.array([[1., 2.], [-1., 3.], [2., 4.]]),
            smri_measures=np.array(["a", "b", "c", "d"]),
            condition_names=np.array(["EO", "EC"]), eeg_freq_centers=np.arange(4),
        )

    def test_joint_permutation_and_one_global_sign(self):
        reference = self.reference
        permutation = np.array([2, 0, 1])
        signs = np.array([-1, 1, -1])
        candidate = dict(reference)
        candidate["shared_scores"] = reference["shared_scores"][:, permutation] * signs
        for key in LOADINGS:
            value = reference[key][permutation]
            candidate[key] = value * signs.reshape((-1,) + (1,) * (value.ndim - 1))
        rows = compare_factors(reference, candidate, reference["shared_scores"], candidate["shared_scores"])
        self.assertEqual([row["matched_factor"] for row in rows], [2, 3, 1])
        self.assertEqual([row["alignment_sign"] for row in rows], [1, -1, -1])
        for row in rows:
            for metric in METRICS:
                self.assertAlmostEqual(row[metric], 1.)
        # A discordant EEG direction must stay negative, rather than be independently flipped.
        candidate["a_shared"][0] *= -1
        rows = compare_factors(reference, candidate, reference["shared_scores"], candidate["shared_scores"])
        self.assertAlmostEqual(rows[2]["eeg_cosine"], -1.)

    def test_full_cohort_orientation_preserves_reconstruction(self):
        candidate, signs = orient_full_cohort(self.reference, "EO")
        np.testing.assert_array_equal(signs, [1, -1, 1])
        for factor in range(3):
            before = np.einsum("n,kfc->nkfc", self.reference["shared_scores"][:, factor], eeg_loading(self.reference, factor))
            after = np.einsum("n,kfc->nkfc", candidate["shared_scores"][:, factor], eeg_loading(candidate, factor))
            np.testing.assert_allclose(before, after)
        self.assertTrue((candidate["c_shared"][:, 0] > 0).all())

    def test_common_block_and_participant_alignment(self):
        candidate = {key: value.copy() for key, value in self.reference.items()}
        candidate["A_shared"] = np.pad(candidate["A_shared"], ((0, 0), (0, 1), (0, 0)), constant_values=1000.)
        candidate["M_shared"] = np.pad(candidate["M_shared"], ((0, 0), (0, 1), (0, 1)), constant_values=1000.)
        order = np.array([6, 1, 9, 2, 0, 11, 7, 3, 8, 4, 10, 5])
        candidate["subject_ids"] = candidate["subject_ids"][order]
        candidate["shared_scores"] = candidate["shared_scores"][order]
        table = compare_fit(self.reference, candidate, common_mri=True, full_cohort=True)
        np.testing.assert_allclose(table[list(METRICS)], 1.)

    def test_quantiles_are_descriptive_per_factor(self):
        details = pd.DataFrame([dict(reference_factor=1, **{key: value for key in METRICS}) for value in (0., .2, .4, .6, .8)])
        summary = participant_summary(details).iloc[0]
        self.assertEqual(summary["n_fits"], 5)
        self.assertAlmostEqual(summary["score_r_q25"], .2)
        self.assertAlmostEqual(summary["score_r_median"], .4)
        self.assertAlmostEqual(summary["score_r_q75"], .6)


class SavedSubsetTest(unittest.TestCase):
    def test_manifest_order_is_used_without_fold_generation(self):
        rows = []
        for repeat in range(1, 6):
            for fold in range(1, 6):
                train = [i for i in reversed(range(10)) if i // 2 != fold - 1]
                rows.extend(dict(repeat=repeat, fold=fold, train_order=order, row_index=i) for order, i in enumerate(train))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.csv"
            pd.DataFrame(rows).sample(frac=1, random_state=4).to_csv(path, index=False)
            groups = read_splits(path, 10)
            np.testing.assert_array_equal(groups[1, 1], np.arange(9, 1, -1))
            malformed = pd.DataFrame(rows)
            malformed.loc[0, "row_index"] = malformed.loc[1, "row_index"]
            malformed.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_splits(path, 10)

    def test_fit_export_uses_shared_inference_and_preserves_input_entries(self):
        rng = np.random.default_rng(10)
        arrays = dict(S_tilde=rng.normal(size=(10, 4, 2)), Sigma_tilde=rng.normal(size=(10, 4, 4)),
                      E=rng.normal(size=(10, 3, 5, 2)), lambda_vals=np.arange(1., 5.), subject_ids=np.array([str(i) for i in range(10)]))
        arrays["Sigma_tilde"] = (arrays["Sigma_tilde"] + arrays["Sigma_tilde"].transpose(0, 2, 1)) / 2
        order = np.array([8, 2, 6, 0, 1, 3, 5, 9])
        data = make_data(arrays, order, 3)
        np.testing.assert_array_equal(data.S_tilde, arrays["S_tilde"][order, :3])
        np.testing.assert_array_equal(data.Sigma_tilde, arrays["Sigma_tilde"][order, :3, :3])
        cfg = asdict(FixedRankGEMFConfig(R=2, Ls=1, Lf=1, Le=1, score_ridge=.1, max_iter=3, verbose=False))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fit"
            fit_once(data, cfg, .1, {}, out)
            with np.load(out / "fit_results.npz", allow_pickle=False) as saved:
                fit = {key: saved[key] for key in saved.files}
            np.testing.assert_array_equal(fit["subject_ids"], arrays["subject_ids"][order])
            np.testing.assert_allclose(fit["shared_scores"], infer_shared_scores(fit, data, 1., 1., 1., ridge=.1), rtol=0, atol=0)
            self.assertFalse(np.allclose(fit["shared_scores"], fit["Z"]))
            self.assertEqual(json.loads(fit["config_json"].item()), cfg)


if __name__ == "__main__":
    unittest.main()
