import numpy as np
import pandas as pd
from gws_core import BaseTestCase, Table, TaskRunner
from gws_design_of_experiments.umap.umap_task import UMAPTask


class TestUMAPTask(BaseTestCase):
    """
    Unit tests for the UMAPTask.

    Note: This task runs UMAP in an isolated Mamba virtual environment.
    The environment must be installed before running these tests
    (handled automatically on first run via UMAPEnvHelper).
    """

    def test_umap_basic(self):
        """Test UMAP dimensionality reduction on a simple synthetic dataset."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
                "f3": np.random.randn(n),
                "f4": np.random.randn(n),
            }
        )

        runner = TaskRunner(
            task_type=UMAPTask,
            inputs={"data": Table(df)},
            params={
                "n_neighbors": 10,
                "min_dist": 0.1,
                "metric": "euclidean",
                "scale_data": True,
            },
        )
        outputs = runner.run()

        self.assertIsNotNone(outputs["umap_2d_plot"])
        self.assertIsNotNone(outputs["umap_3d_plot"])
        self.assertIsNotNone(outputs["umap_2d_table"])
        self.assertIsNotNone(outputs["umap_3d_table"])

        # 2D table should have UMAP1 and UMAP2 columns
        umap_2d_df = outputs["umap_2d_table"].get_data()
        self.assertIn("UMAP1", umap_2d_df.columns)
        self.assertIn("UMAP2", umap_2d_df.columns)
        self.assertEqual(len(umap_2d_df), n)

        # 3D table should have UMAP1, UMAP2 and UMAP3 columns
        umap_3d_df = outputs["umap_3d_table"].get_data()
        self.assertIn("UMAP1", umap_3d_df.columns)
        self.assertIn("UMAP2", umap_3d_df.columns)
        self.assertIn("UMAP3", umap_3d_df.columns)
        self.assertEqual(len(umap_3d_df), n)

    def test_umap_with_clustering(self):
        """Test UMAP with K-Means clustering enabled."""
        np.random.seed(7)
        n = 60
        # Create two well-separated clusters
        cluster_a = pd.DataFrame(
            {
                "f1": np.random.randn(n // 2) + 5,
                "f2": np.random.randn(n // 2) + 5,
                "f3": np.random.randn(n // 2),
            }
        )
        cluster_b = pd.DataFrame(
            {
                "f1": np.random.randn(n // 2) - 5,
                "f2": np.random.randn(n // 2) - 5,
                "f3": np.random.randn(n // 2),
            }
        )
        df = pd.concat([cluster_a, cluster_b], ignore_index=True)

        runner = TaskRunner(
            task_type=UMAPTask,
            inputs={"data": Table(df)},
            params={
                "n_neighbors": 10,
                "min_dist": 0.1,
                "metric": "euclidean",
                "scale_data": True,
                "n_clusters": 2,
            },
        )
        outputs = runner.run()

        umap_2d_df = outputs["umap_2d_table"].get_data()
        # Cluster column should be present when n_clusters is specified
        self.assertIn("Cluster", umap_2d_df.columns)
        # Exactly 2 unique cluster labels
        self.assertEqual(umap_2d_df["Cluster"].nunique(), 2)

    def test_umap_columns_to_exclude(self):
        """Test that columns_to_exclude removes specified columns from analysis."""
        np.random.seed(3)
        n = 40
        df = pd.DataFrame(
            {
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
                "f3": np.random.randn(n),
                "sample_id": np.arange(n, dtype=float),  # Should be excluded
            }
        )

        runner = TaskRunner(
            task_type=UMAPTask,
            inputs={"data": Table(df)},
            params={
                "n_neighbors": 5,
                "min_dist": 0.1,
                "metric": "euclidean",
                "scale_data": True,
                "columns_to_exclude": ["sample_id"],
            },
        )
        outputs = runner.run()

        umap_2d_df = outputs["umap_2d_table"].get_data()
        self.assertEqual(len(umap_2d_df), n)
