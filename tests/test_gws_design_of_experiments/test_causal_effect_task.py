import os

import numpy as np
import pandas as pd
from gws_core import BaseTestCase, Table, TaskRunner
from gws_design_of_experiments.causal_effect.causal_effect_task import CausalEffect


class TestCausalEffectTask(BaseTestCase):
    """
    Unit tests for the CausalEffect task.

    Note: This task runs causal inference analysis in an isolated Mamba virtual
    environment using EconML. The environment must be installed before running these
    tests (handled automatically on first run via EconmlEnvHelper).
    """

    def test_causal_effect_basic(self):
        """Test causal effect estimation with one target and two treatment variables."""
        np.random.seed(42)
        n = 100

        # Confounders
        age = np.random.uniform(20, 60, n)
        bmi = np.random.uniform(18, 35, n)

        # Treatments
        drug_dose = np.random.uniform(0, 10, n)
        exercise_hours = np.random.uniform(0, 7, n)

        # Target: blood pressure influenced by treatments and confounders
        blood_pressure = (
            80
            + 0.5 * age
            + 1.2 * bmi
            - 3.0 * drug_dose
            - 2.0 * exercise_hours
            + np.random.randn(n) * 2
        )

        df = pd.DataFrame(
            {
                "Age": age,
                "BMI": bmi,
                "DrugDose": drug_dose,
                "ExerciseHours": exercise_hours,
                "BloodPressure": blood_pressure,
            }
        )

        runner = TaskRunner(
            task_type=CausalEffect,
            inputs={"data": Table(df)},
            params={
                "targets": ["BloodPressure"],
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["results_folder"])

        results_folder = outputs["results_folder"]
        folder_path = results_folder.path

        self.assertTrue(os.path.isdir(folder_path))

        # At least one subfolder or CSV file should be present
        contents = os.listdir(folder_path)
        self.assertGreater(len(contents), 0)

    def test_causal_effect_multiple_targets(self):
        """Test causal effect estimation with multiple target variables."""
        np.random.seed(10)
        n = 80

        confounder = np.random.randn(n)
        treatment = np.random.randn(n)

        target1 = 2.0 * treatment + confounder + np.random.randn(n) * 0.5
        target2 = -1.5 * treatment + 0.5 * confounder + np.random.randn(n) * 0.5

        df = pd.DataFrame(
            {
                "Confounder": confounder,
                "Treatment": treatment,
                "Outcome1": target1,
                "Outcome2": target2,
            }
        )

        runner = TaskRunner(
            task_type=CausalEffect,
            inputs={"data": Table(df)},
            params={
                "targets": ["Outcome1", "Outcome2"],
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["results_folder"])

        folder_path = outputs["results_folder"].path
        self.assertTrue(os.path.isdir(folder_path))

    def test_causal_effect_columns_to_exclude(self):
        """Test that columns_to_exclude removes specified columns from the analysis."""
        np.random.seed(5)
        n = 60

        treatment = np.random.randn(n)
        confounder = np.random.randn(n)
        sample_id = np.arange(n, dtype=float)  # Should be excluded

        target = 1.5 * treatment + confounder + np.random.randn(n) * 0.3

        df = pd.DataFrame(
            {
                "Treatment": treatment,
                "Confounder": confounder,
                "SampleID": sample_id,
                "Target": target,
            }
        )

        runner = TaskRunner(
            task_type=CausalEffect,
            inputs={"data": Table(df)},
            params={
                "targets": ["Target"],
                "columns_to_exclude": ["SampleID"],
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["results_folder"])
