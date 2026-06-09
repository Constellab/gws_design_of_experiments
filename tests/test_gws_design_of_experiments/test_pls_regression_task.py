import numpy as np
import pandas as pd
from gws_core import BaseTestCase, Table, TaskRunner
from gws_design_of_experiments.pls.pls_regression_task import PLSRegressorTask


class TestPLSRegressionTask(BaseTestCase):
    """Unit tests for the PLSRegressorTask."""

    def test_pls_single_target(self):
        """Test PLS regression with a single target variable."""
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
                "y": 2.0 * x1 + 3.0 * x2 + np.random.randn(n) * 0.1,
            }
        )

        runner = TaskRunner(
            task_type=PLSRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": ["y"],
                "test_size": 0.2,
                "scale_data": True,
            },
        )
        outputs = runner.run()

        # All outputs must be present
        self.assertIsNotNone(outputs["summary_table"])
        self.assertIsNotNone(outputs["vip_table"])
        self.assertIsNotNone(outputs["plot_components"])
        self.assertIsNotNone(outputs["vip_plot"])
        self.assertIsNotNone(outputs["plot_train_set"])
        self.assertIsNotNone(outputs["plot_test_set"])

        # Summary table must contain R2 and RMSE columns
        summary_df = outputs["summary_table"].get_data()
        self.assertIn("R2", summary_df.columns)
        self.assertIn("RMSE", summary_df.columns)
        self.assertIn("split", summary_df.columns)

        # VIP table must have one row per feature
        vip_df = outputs["vip_table"].get_data()
        self.assertEqual(len(vip_df), 3)  # x1, x2, x3

    def test_pls_multi_target(self):
        """Test PLS regression with multiple target variables."""
        np.random.seed(0)
        n = 80
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "y1": 2.0 * x1 - x3 + np.random.randn(n) * 0.05,
                "y2": x2 + 0.5 * x3 + np.random.randn(n) * 0.05,
            }
        )

        runner = TaskRunner(
            task_type=PLSRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": ["y1", "y2"],
                "test_size": 0.2,
                "scale_data": True,
            },
        )
        outputs = runner.run()

        summary_df = outputs["summary_table"].get_data()
        # Both targets and both splits (train/test) should appear
        self.assertIn("y1", summary_df["target"].values)
        self.assertIn("y2", summary_df["target"].values)

    def test_pls_columns_to_exclude(self):
        """Test that columns_to_exclude properly removes specified columns."""
        np.random.seed(7)
        n = 60
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "id": np.arange(n),  # Should be excluded
                "y": 1.5 * x1 + np.random.randn(n) * 0.1,
            }
        )

        runner = TaskRunner(
            task_type=PLSRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": ["y"],
                "columns_to_exclude": ["id"],
                "test_size": 0.2,
                "scale_data": True,
            },
        )
        outputs = runner.run()

        vip_df = outputs["vip_table"].get_data()
        feature_names = (
            vip_df.iloc[:, 0].tolist() if "Feature" in vip_df.columns else vip_df.index.tolist()
        )
        # 'id' column must not appear in VIP table
        self.assertNotIn("id", str(feature_names))

    def test_pls_without_scaling(self):
        """Test PLS regression with data scaling disabled."""
        np.random.seed(1)
        n = 60
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "y": 3.0 * x1 - x2 + np.random.randn(n) * 0.1,
            }
        )

        runner = TaskRunner(
            task_type=PLSRegressorTask,
            inputs={"data": Table(df)},
            params={
                "target": ["y"],
                "test_size": 0.2,
                "scale_data": False,
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["summary_table"])
        self.assertIsNotNone(outputs["vip_table"])
        summary_df = outputs["summary_table"].get_data()
        self.assertIn("R2", summary_df.columns)
        # Both train and test splits must appear
        self.assertIn("train", summary_df["split"].values)
        self.assertIn("test", summary_df["split"].values)
