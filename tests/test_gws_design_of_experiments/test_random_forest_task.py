import numpy as np
import pandas as pd
from gws_core import BaseTestCase, Table, TaskRunner
from gws_design_of_experiments.random_forest.random_forest_task import RandomForestRegressorTask


class TestRandomForestRegressorTask(BaseTestCase):
    """Unit tests for the RandomForestRegressorTask."""

    def test_random_forest_basic(self):
        """Test Random Forest regression with a single target variable."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "y": 2.0 * x1 + 3.0 * x2 + np.random.randn(n) * 0.5,
            }
        )

        runner = TaskRunner(
            task_type=RandomForestRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": "y",
                "test_size": 0.2,
            },
        )
        outputs = runner.run()

        # All outputs must be present
        self.assertIsNotNone(outputs["summary_table"])
        self.assertIsNotNone(outputs["vip_table"])
        self.assertIsNotNone(outputs["plot_estimators"])
        self.assertIsNotNone(outputs["vip_plot"])
        self.assertIsNotNone(outputs["plot_train_set"])
        self.assertIsNotNone(outputs["plot_test_set"])

        # Summary table must contain expected columns
        summary_df = outputs["summary_table"].get_data()
        self.assertIn("R2", summary_df.columns)
        self.assertIn("RMSE", summary_df.columns)
        self.assertIn("split", summary_df.columns)

        # Both train and test splits must appear
        self.assertIn("train", summary_df["split"].values)
        self.assertIn("test", summary_df["split"].values)

        # VIP table must have one row per feature
        vip_df = outputs["vip_table"].get_data()
        self.assertEqual(len(vip_df), 3)  # x1, x2, x3

    def test_random_forest_columns_to_exclude(self):
        """Test that columns_to_exclude removes specified columns from analysis."""
        np.random.seed(10)
        n = 80
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "sample_id": np.arange(n),  # Should be excluded
                "y": 1.5 * x1 + np.random.randn(n) * 0.2,
            }
        )

        runner = TaskRunner(
            task_type=RandomForestRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": "y",
                "columns_to_exclude": ["sample_id"],
                "test_size": 0.2,
            },
        )
        outputs = runner.run()

        # Only x1 and x2 should appear in the VIP table
        vip_df = outputs["vip_table"].get_data()
        self.assertEqual(len(vip_df), 2)

    def test_random_forest_large_test_split(self):
        """Test Random Forest with a larger test split (0.3)."""
        np.random.seed(99)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "y": 2.0 * x1 + x2 + np.random.randn(n) * 0.5,
            }
        )

        runner = TaskRunner(
            task_type=RandomForestRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": "y",
                "test_size": 0.3,
            },
        )
        outputs = runner.run()

        summary_df = outputs["summary_table"].get_data()
        # Both splits must appear
        self.assertIn("train", summary_df["split"].values)
        self.assertIn("test", summary_df["split"].values)

    def test_random_forest_r2_on_structured_data(self):
        """Test that R² on training set is positive for a well-structured dataset."""
        np.random.seed(5)
        n = 120
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "y": 5.0 * x1 - 2.0 * x2 + np.random.randn(n) * 0.01,
            }
        )

        runner = TaskRunner(
            task_type=RandomForestRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": "y",
                "test_size": 0.2,
            },
        )
        outputs = runner.run()

        summary_df = outputs["summary_table"].get_data()
        train_r2 = summary_df.loc[summary_df["split"] == "train", "R2"].values[0]
        self.assertGreater(train_r2, 0.5)
