"""Check paired denominators and uncertainty in the paper/debiasing reports."""
import math
import unittest

import numpy as np
import pandas as pd

from summarize_cross_u_benchmark import common_defined, paired_differences, DEBIASED, PLUGIN, RTI, DELTA
from make_cross_u_paper_tables import summarize_tables


class CrossUReportingTests(unittest.TestCase):
    def test_undefined_delta_is_removed_from_every_matched_method(self):
        rows=[]
        for seed in (1,2):
            for method in (DEBIASED,RTI,DELTA):
                rows.append({"N":250,"eps_smoothness":.75,"rho":.1,"random_key":seed,
                             "method":method,"coverage":np.nan if method==DELTA and seed==2 else True})
        frame=pd.DataFrame(rows)
        result=common_defined(frame,[DEBIASED,RTI,DELTA])
        self.assertEqual(len(result),3)
        self.assertEqual(set(result.random_key),{1})
        self.assertEqual(frame.coverage.isna().sum(),1)

    def test_paired_uncertainty_uses_within_run_differences(self):
        rows=[]
        for seed,(first,second) in enumerate([(1,0),(0,1),(1,1)]):
            for method,value in ((DEBIASED,first),(PLUGIN,second)):
                rows.append({"N":250,"eps_smoothness":.75,"rho":.1,"random_key":seed,
                             "method":method,"coverage":value})
        result=paired_differences(pd.DataFrame(rows),DEBIASED,PLUGIN,["coverage"]).iloc[0]
        self.assertEqual(result.paired_runs,3)
        self.assertEqual(result.difference,0)
        self.assertAlmostEqual(result.mc_standard_error,1/math.sqrt(3))
        self.assertEqual(result.first_only_successes,1)
        self.assertEqual(result.second_only_successes,1)

    def test_missing_pair_is_counted_without_imputation(self):
        rows=[]
        for seed,(first,second) in enumerate([(1,0),(1,np.nan)]):
            for method,value in ((DEBIASED,first),(DELTA,second)):
                rows.append({"N":250,"eps_smoothness":.75,"rho":.1,"random_key":seed,
                             "method":method,"coverage":value})
        result=paired_differences(pd.DataFrame(rows),DEBIASED,DELTA,["coverage"]).iloc[0]
        self.assertEqual(result.available_pairs,2)
        self.assertEqual(result.paired_runs,1)
        self.assertEqual(result.undefined_pairs,1)
        self.assertEqual(result.difference,1)
        self.assertTrue(math.isnan(result.mc_standard_error))

    def test_zero_inclusion_handles_empty_and_undefined_sets(self):
        rows = []
        # One empty set, one containing zero, and one strictly positive interval.
        for seed, (lower, upper, width) in enumerate([(0., -1., 0.), (0., 4., 2.), (1., 9., 2.)]):
            for method in (DEBIASED, RTI, DELTA):
                undefined = method == DELTA and seed == 0
                rows.append({
                    "N": 250, "eps_smoothness": .5, "rho": .1, "random_key": seed,
                    "method": method, "population_hsic": 1.,
                    "coverage": np.nan if undefined else upper >= 1.,
                    "lower_hsic": np.nan if undefined else lower,
                    "upper_hsic": np.nan if undefined else upper,
                    "empty": np.nan if undefined else upper < lower,
                    "width_sqrt_hsic": np.nan if undefined else width,
                    "ci_excludes_zero": np.nan if undefined else upper < 0 or lower > 0,
                })
        frame = pd.DataFrame(rows)
        with self.assertRaisesRegex(ValueError, "refusing to discard"):
            summarize_tables(frame)
        # Replace the undefined delta with a valid squared-U interval mapping to [0, 2].
        mask = frame.method.eq(DELTA) & frame.random_key.eq(0)
        frame.loc[mask, ["coverage", "lower_hsic", "upper_hsic", "empty",
                         "width_sqrt_hsic", "ci_excludes_zero"]] = [True, 0., 4., False, 2., False]
        full = summarize_tables(frame).set_index("method")
        self.assertTrue(full.reported_runs.eq(3).all())
        self.assertAlmostEqual(full.loc[DEBIASED, "mean_width_sqrt_hsic"], 4 / 3)
        self.assertAlmostEqual(full.loc[DEBIASED, "probability_includes_zero"], 1 / 3)
        self.assertEqual(full.loc[DEBIASED, "empty_sets"], 1)
        self.assertAlmostEqual(full.loc[DELTA, "probability_includes_zero"], 2 / 3)


if __name__=="__main__":
    unittest.main()
