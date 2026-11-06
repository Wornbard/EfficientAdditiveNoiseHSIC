"""Reporting checks for robust covariate intervals and classical experiments."""
from pathlib import Path
from statistics import NormalDist
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

import summarize_toy_additive_noise_selected_1000 as classical


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = nbformat.read(Path(__file__).with_name("submission_with_covariate.ipynb"), as_version=4)
        cls.covariate = {"np": np, "NormalDist": NormalDist}
        # Definition-only cells; no simulations or figure writes.
        exec(notebook.cells[2].source, cls.covariate)
        exec(notebook.cells[8].source, cls.covariate)

    def test_squared_interval_uses_unclipped_u_and_score_variance(self):
        for contributions in (np.array([1.0, -0.5, 0.25]), np.array([1.0, 2.0, 3.0])):
            gram = np.outer(contributions, contributions)
            ci = self.covariate["delta_sqrt_hsic_ci_from_blocks"]([[gram]], 1.96)
            u = (contributions.sum()**2 - (contributions**2).sum()) / 6
            scores = contributions * (contributions.sum() - contributions) / 2
            radius = 1.96 * 2 * np.sqrt(np.var(scores, ddof=1) / 3)
            self.assertAlmostEqual(ci["hsic_center"], u)
            self.assertAlmostEqual(ci["hsic_upper"], u + radius)
            self.assertAlmostEqual(ci["sqrt_upper"], np.sqrt(u + radius))
            self.assertAlmostEqual(ci["sqrt_lower"], np.sqrt(max(u - radius, 0)))
            self.assertFalse(ci["empty"])

    def test_empty_and_zero_singleton_are_distinct(self):
        interval = self.covariate["delta_sqrt_hsic_ci_from_moments"]
        empty = interval(-1.0, 0.0, 3, 1.96)
        singleton = interval(0.0, 0.0, 3, 1.96)
        self.assertTrue(empty["empty"])
        self.assertTrue(np.isnan(empty["sqrt_upper"]))
        self.assertEqual(empty["sqrt_width"], 0.0)
        self.assertFalse(singleton["empty"])
        self.assertEqual((singleton["sqrt_lower"], singleton["sqrt_upper"]), (0.0, 0.0))

    def test_nonfinite_moments_stop_instead_of_dropping_repetition(self):
        with self.assertRaises(ValueError):
            self.covariate["delta_sqrt_hsic_ci_from_blocks"]([[np.full((3, 3), np.nan)]], 1.96)

    def test_all_repetitions_including_empty_enter_coverage_and_width(self):
        records = []
        for level in self.covariate["DELTA_NOMINAL_LEVELS"]:
            for estimator in ("plug-in", "debiased"):
                for empty in (False, True):
                    records.append(dict(nominal_level=level, estimator=estimator,
                                        finite=not empty, empty=empty, covered=not empty,
                                        clipped_center=empty, sqrt_width=0.0 if empty else 0.2,
                                        sqrt_lower=np.nan if empty else 0.0,
                                        sqrt_upper=np.nan if empty else 0.2,
                                        sqrt_center=0.1, true_sqrt=0.1))
        for row in self.covariate["summarize_delta_records"](records):
            self.assertEqual((row["n"], row["n_finite"], row["n_empty"]), (2, 1, 1))
            self.assertEqual(row["coverage_all"], 0.5)
            self.assertEqual(row["coverage_finite"], 1.0)
            self.assertEqual(row["mean_width"], 0.1)
        records[0]["sqrt_width"] = np.nan
        with self.assertRaises(ValueError):
            self.covariate["summarize_delta_records"](records)

    def test_fresh_classical_table_can_be_regenerated_without_selection_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["CANONICAL_SUMMARY_CSV", "CANONICAL_WIDE_CSV",
                     "CANONICAL_PUBLICATION_AUDIT_CSV", "CANONICAL_MARKDOWN", "CANONICAL_LATEX"]
            paths = {name: root / (name + ".csv") for name in names}
            pd.DataFrame({"source_repetition_start": [100] * 32,
                          "source_repetition_end": [1099] * 32}).to_csv(paths[names[0]], index=False)
            with patch.multiple(classical, ROOT=root, **paths):
                self.assertEqual(classical.preserve_n100_canonical_artifacts(), {})


if __name__ == "__main__":
    unittest.main()
