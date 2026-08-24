import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas

import generate_paper_outputs as paper_outputs


class PrefitImprovementStageTest(unittest.TestCase):
    def test_load_results_joins_separate_prefit_table_and_uses_nobs_as_count(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = Path(tempdir)
            fitted_path = directory / "fitted.gz"
            prefit_path = directory / "prefits.gz"
            pandas.DataFrame(
                {
                    "system_id": ["system-a"],
                    "baseline": [5 * 365],
                    "nobs": [7],
                    "RV accuracy": [100.0],
                    "Macc": [8.0],
                    "a": [2.0],
                    "Porb": [500.0],
                    "b": [0.1],
                    "fitted_Macc": [8.1],
                    "fitted_a": [2.1],
                    "fitted_porb": [510.0],
                    "fitted_b": [0.12],
                }
            ).to_pickle(fitted_path)
            pandas.DataFrame(
                {
                    "simulation_id": ["system-a"],
                    "prefit_fitted_Macc": [7.5],
                    "prefit_fitted_a": [1.8],
                }
            ).to_pickle(prefit_path)
            previous = os.environ.get("SLRV_PREFIT_OUTPUT_PATH")
            os.environ["SLRV_PREFIT_OUTPUT_PATH"] = str(prefit_path)
            try:
                results = paper_outputs.load_results(fitted_path)
            finally:
                if previous is None:
                    os.environ.pop("SLRV_PREFIT_OUTPUT_PATH", None)
                else:
                    os.environ["SLRV_PREFIT_OUTPUT_PATH"] = previous

        self.assertEqual(results.loc[0, "n_rv"], 7)
        self.assertEqual(results.loc[0, "prefit_fitted_Macc"], 7.5)
        self.assertEqual(results.loc[0, "strategy"], "5 yr, Nobs=7")

    def test_prefit_error_rows_include_rv_no_peak_time_stage(self):
        results = pandas.DataFrame(
            {
                "system_id": ["a", "b"],
                "n_rv": [3, 10],
                "Macc": [5.0, 10.0],
                "a": [1.0, 2.0],
                "Porb": [100.0, 200.0],
                "b": [0.1, 0.2],
                "prefit_fitted_Macc": [4.5, 9.0],
                "fitted_Macc": [5.1, 10.1],
                "rvfit_fitted_Macc": [5.2, 10.2],
                "rvnoprior_fitted_Macc": [6.0, 12.0],
            }
        )

        rows = paper_outputs.prefit_joint_error_rows(results)
        mass_rows = rows[rows["parameter"] == "Macc"]

        self.assertEqual(
            set(mass_rows["fit_stage"]),
            {
                "Photometry only",
                "Photometry + RV",
                "RV only",
                "RV only, no peak time",
            },
        )
        self.assertIn(
            "RV only, no peak time",
            paper_outputs.PREFIT_COMPARISON_STAGE_STYLES,
        )

    def test_prefit_class_rows_include_rv_no_peak_time_stage(self):
        results = pandas.DataFrame(
            {
                "system_id": ["a", "b"],
                "n_rv": [3, 10],
                "Macc": [4.0, 8.0],
                "prefit_fitted_Macc": [2.5, 6.0],
                "fitted_Macc": [4.2, 8.2],
                "rvfit_fitted_Macc": [4.1, 8.1],
                "rvnoprior_fitted_Macc": [5.0, 12.0],
            }
        )

        rows = paper_outputs.prefit_compact_object_class_rows(results)

        self.assertIn("RV only, no peak time", set(rows["fit_stage"]))


if __name__ == "__main__":
    unittest.main()
