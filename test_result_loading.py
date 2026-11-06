"""Cache reuse must not depend on how the same archive path is spelled."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
from load_experiment_results import load_experiment_rows

class ResultsCacheTests(unittest.TestCase):
    def test_filtered_request_reuses_valid_full_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "cfg_0000"
            run.mkdir()
            (run / "results.yaml").write_text("config:\n  N: 250\n  rho: 0.3\n")
            load_experiment_rows(tmp, include_raw=False)
            with patch("load_experiment_results.yaml.safe_load", side_effect=AssertionError("Use full cache")):
                rows = load_experiment_rows(tmp, include_raw=False, keep_columns=["N"], write_cache=False)
            self.assertEqual(rows[0]["N"], 250)
            self.assertNotIn("rho", rows[0])
            self.assertEqual(Path(rows[0]["results_path"]), run / "results.yaml")

    def test_relative_absolute_paths_and_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment=Path(tmp)/"experiment"
            run=experiment/"cfg_0000"
            run.mkdir(parents=True)
            result=run/"results.yaml"
            result.write_text("config:\n  N: 250\n")
            load_experiment_rows(os.path.relpath(experiment),include_raw=False)
            with patch("load_experiment_results.yaml.safe_load",side_effect=AssertionError("Cache should be reused")):
                rows=load_experiment_rows(experiment,include_raw=False,write_cache=False)
            self.assertEqual(rows[0]["N"],250)
            self.assertEqual(Path(rows[0]["results_path"]),result)
            result.write_text("config:\n  N: 1000\n")
            rows=load_experiment_rows(experiment,include_raw=False,write_cache=False)
            self.assertEqual(rows[0]["N"],1000)

if __name__ == "__main__":
    unittest.main()
