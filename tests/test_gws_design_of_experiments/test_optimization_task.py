import os

import numpy as np
import pandas as pd
from gws_core import BaseTestCase, JSONDict, Table, TaskRunner
from gws_design_of_experiments.optimization.optimization_task import Optimization


class TestOptimizationTask(BaseTestCase):
    """
    Unit tests for the Optimization task.

    Note: This task runs the optimization algorithm in an isolated Mamba virtual
    environment. The environment must be installed before running these tests
    (handled automatically on first run via OptimizationEnvHelper).
    """

    def test_optimization_basic(self):
        """Test optimization on a simple synthetic dataset with one target."""
        np.random.seed(42)
        n = 30

        temperature = np.random.uniform(20, 80, n)
        pressure = np.random.uniform(1, 10, n)
        ph = np.random.uniform(5, 9, n)

        # Simulated yield: higher temperature and pH improve yield
        yield_ = 0.5 * temperature + 2.0 * ph - 0.3 * pressure + np.random.randn(n) * 2

        df = pd.DataFrame(
            {
                "Temperature": temperature,
                "Pressure": pressure,
                "pH": ph,
                "Yield": yield_,
            }
        )

        # Manual constraints: bounds on input features
        manual_constraints = JSONDict(
            {
                "Temperature": {"lower_bound": 20, "upper_bound": 80},
                "Pressure": {"lower_bound": 1, "upper_bound": 10},
                "pH": {"lower_bound": 5, "upper_bound": 9},
            }
        )

        runner = TaskRunner(
            task_type=Optimization,
            inputs={
                "data": Table(df),
                "manual_constraints": manual_constraints,
            },
            params={
                "population_size": 10,
                "iterations": 3,
                "targets_thresholds": [
                    {"targets": "Yield", "thresholds": 50},
                ],
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["results_folder"])

        # The results folder should contain the expected CSV output files
        results_folder = outputs["results_folder"]
        folder_path = results_folder.path

        self.assertTrue(os.path.isdir(folder_path))

        files_in_folder = os.listdir(folder_path)
        self.assertGreater(len(files_in_folder), 0)

        # Check that key output files are present
        expected_files = [
            "generalized_solutions.csv",
            "best_generalized_solution.csv",
        ]
        for expected_file in expected_files:
            self.assertIn(
                expected_file,
                files_in_folder,
                msg=f"Expected output file '{expected_file}' not found in results folder",
            )

    def test_optimization_columns_to_exclude(self):
        """Test that columns_to_exclude removes specified columns from the optimization."""
        np.random.seed(1)
        n = 30

        temperature = np.random.uniform(30, 70, n)
        pressure = np.random.uniform(2, 8, n)
        sample_id = np.arange(n, dtype=float)

        yield_ = 0.4 * temperature - 0.1 * pressure + np.random.randn(n) * 1

        df = pd.DataFrame(
            {
                "Temperature": temperature,
                "Pressure": pressure,
                "SampleID": sample_id,  # Should be excluded
                "Yield": yield_,
            }
        )

        manual_constraints = JSONDict(
            {
                "Temperature": {"lower_bound": 30, "upper_bound": 70},
                "Pressure": {"lower_bound": 2, "upper_bound": 8},
            }
        )

        runner = TaskRunner(
            task_type=Optimization,
            inputs={
                "data": Table(df),
                "manual_constraints": manual_constraints,
            },
            params={
                "population_size": 10,
                "iterations": 3,
                "columns_to_exclude": ["SampleID"],
                "targets_thresholds": [
                    {"targets": "Yield", "thresholds": 30},
                ],
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["results_folder"])
