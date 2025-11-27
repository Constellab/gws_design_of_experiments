from umap import UMAP
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from gws_core import (ConfigParams, ConfigSpecs, InputSpec, InputSpecs,
                      OutputSpec, OutputSpecs, ListParam,
                      Table, Task, TaskInputs, TaskOutputs, TypingStyle, PlotlyResource,
                      task_decorator, IntParam, FloatParam, BoolParam, StrParam)


@task_decorator("UMAPTask", human_name="UMAP Dimensionality Reduction",
                short_description="UMAP for dimensionality reduction and visualization",
                style=TypingStyle.material_icon(material_icon_name="scatter_plot",
                                                background_color="#85c5fe"))
class UMAPTask(Task):
    """
    Performs UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction.

    This task reduces high-dimensional data to 2D or 3D for visualization and
    optionally performs clustering to identify groups in the data.

    The task performs the following steps:
    1. Optionally scales the data using StandardScaler
    2. Applies UMAP dimensionality reduction
    3. Optionally performs K-Means clustering on the UMAP embedding
    4. Generates interactive visualizations

    Inputs:
        - data: Table containing the features to reduce

    Outputs:
        - umap_plot: Interactive plot of UMAP embedding with optional clusters
        - umap_table: Table containing UMAP coordinates and cluster assignments

    Configuration:
        - n_neighbors: Number of neighbors for UMAP (controls local vs global structure)
        - min_dist: Minimum distance between points in low-dimensional space
        - metric: Distance metric to use
        - scale_data: Whether to standardize features before UMAP
        - n_clusters: Number of clusters for K-Means (optional)
        - color_by: Column name to color points by (optional)
        - columns_to_exclude: Comma-separated list of column names to exclude from UMAP analysis
    """
    METRICS_OPTION = ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra",
                        "braycurtis", "mahalanobis", "wminkowski", "seuclidean",
                        "cosine", "correlation", "haversine",  "hamming",
                        "jaccard", "dice", "russelrao", "kulsinski",
                        "ll_dirichlet", "hellinger", "rogerstanimoto", "sokalmichener",
                        "sokalsneath", "yule"]
    input_specs = InputSpecs({'data': InputSpec(Table, human_name="Data",
                             short_description="Input data for UMAP")})

    config_specs = ConfigSpecs({
        'n_neighbors': IntParam(
            default_value=15,
            min_value=2,
            max_value=200,
            human_name="Number of Neighbors",
            short_description="Controls how UMAP balances local vs global structure"),
        'min_dist': FloatParam(
            default_value=0.1,
            min_value=0.0,
            max_value=0.99,
            human_name="Minimum Distance",
            short_description="Minimum distance between points in the embedding"),
        'metric': StrParam(
            default_value='euclidean',
            allowed_values=METRICS_OPTION,
            human_name="Distance Metric",
            short_description="Distance metric to use (euclidean, manhattan, cosine, etc.)"),
        'scale_data': BoolParam(
            default_value=True,
            human_name="Scale Data",
            short_description="Whether to scale the data before applying UMAP"),
        'n_clusters': IntParam(
            min_value=2,
            human_name="Number of Clusters",
            short_description="Number of clusters for K-Means clustering (optional)",
            optional=True),
        'color_by': StrParam(
            human_name="Color By Column",
            short_description="Column name to color points by (optional)",
            optional=True),
        'columns_to_exclude': ListParam(
            human_name="Columns to Exclude",
            short_description="List of column names to exclude from UMAP analysis",
            optional=True),
        'hover_data_columns': ListParam(
            human_name="Hover Data Columns",
            short_description="List of column names to display as metadata on hover",
            optional=True)
    })

    output_specs = OutputSpecs({
        'umap_2d_plot': OutputSpec(
            PlotlyResource,
            human_name="UMAP 2D Plot",
            short_description="Interactive UMAP 2D embedding visualization"),
        'umap_3d_plot': OutputSpec(
            PlotlyResource,
            human_name="UMAP 3D Plot",
            short_description="Interactive UMAP 3D embedding visualization"),
        'umap_2d_table': OutputSpec(
            Table,
            human_name="UMAP 2D Table",
            short_description="Table with UMAP 2D coordinates and cluster assignments"),
        'umap_3d_table': OutputSpec(
            Table,
            human_name="UMAP 3D Table",
            short_description="Table with UMAP 3D coordinates and cluster assignments")
    })

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        # Load data
        df = inputs["data"].get_data()

        # Extract hover data columns if specified
        hover_data = {}
        if params['hover_data_columns']:
            hover_cols = [col.strip() for col in params['hover_data_columns']]
            # Validate that columns exist
            invalid_cols = [col for col in hover_cols if col not in df.columns]
            if invalid_cols:
                raise ValueError(f"Hover data columns not found in data: {', '.join(invalid_cols)}")

            for col in hover_cols:
                hover_data[col] = df[col].copy()

        # Exclude specified columns
        columns_to_drop = []
        if params['columns_to_exclude']:
            columns_to_drop = [col.strip() for col in params['columns_to_exclude']]
            # Validate that columns exist
            invalid_cols = [col for col in columns_to_drop if col not in df.columns]
            if invalid_cols:
                raise ValueError(f"Columns not found in data: {', '.join(invalid_cols)}")
            df = df.drop(columns=columns_to_drop)

        # Separate color column if specified
        color_column = None
        if params['color_by'] and params['color_by'] in df.columns:
            color_column = df[params['color_by']].copy()
            X = df.drop(columns=[params['color_by']])
        else:
            X = df.copy()

        # Check for categorical variables
        if X.select_dtypes(include=['object', 'category', 'string']).shape[1] > 0:
            raise ValueError("Categorical variables are not supported. Please encode or remove them or use them to color the points.")

        # Store original index
        original_index = X.index

        # Scaling if requested
        if params['scale_data']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Apply UMAP for 2D
        reducer_2d = UMAP(
            n_components=2,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            random_state=42,
            metric=params['metric']
        )
        embedding_2d = reducer_2d.fit_transform(X_scaled)

        # Apply UMAP for 3D
        reducer_3d = UMAP(
            n_components=3,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            random_state=42,
            metric=params['metric']
        )
        embedding_3d = reducer_3d.fit_transform(X_scaled)

        # Create 2D result dataframe
        result_2d_df = pd.DataFrame({
            'UMAP1': embedding_2d[:, 0],
            'UMAP2': embedding_2d[:, 1]
        }, index=original_index)

        # Create 3D result dataframe
        result_3d_df = pd.DataFrame({
            'UMAP1': embedding_3d[:, 0],
            'UMAP2': embedding_3d[:, 1],
            'UMAP3': embedding_3d[:, 2]
        }, index=original_index)

        # Perform clustering if requested (on 2D embedding)
        cluster_labels = None
        if params['n_clusters'] is not None:
            kmeans = KMeans(n_clusters=params['n_clusters'], random_state=42)
            cluster_labels = kmeans.fit_predict(embedding_2d)
            result_2d_df['Cluster'] = cluster_labels.astype(str)
            result_3d_df['Cluster'] = cluster_labels.astype(str)

        # Add color column if provided
        if color_column is not None:
            result_2d_df['ColorBy'] = color_column.values.astype(str)
            result_3d_df['ColorBy'] = color_column.values.astype(str)

        # Add hover data columns
        for col_name, col_data in hover_data.items():
            result_2d_df[col_name] = col_data.values
            result_3d_df[col_name] = col_data.values

        # Determine color parameter
        color_param = None
        color_labels = {}
        if params['n_clusters'] is not None:
            color_param = 'Cluster'
        elif color_column is not None:
            color_param = 'ColorBy'
            if params['color_by']:
                color_labels = {'ColorBy': params['color_by']}

        # Prepare hover data list for plots
        hover_data_list = list(hover_data.keys()) if hover_data else None

        # Create 2D visualization with px.scatter
        fig_2d = px.scatter(
            result_2d_df,
            x='UMAP1',
            y='UMAP2',
            color=color_param,
            hover_name=result_2d_df.index,
            hover_data=hover_data_list,
            labels=color_labels,
            title="UMAP 2D Embedding"
        )

        fig_2d.update_traces(marker=dict(size=8))
        fig_2d.update_layout(hovermode='closest')

        # Create 3D visualization with px.scatter_3d
        fig_3d = px.scatter_3d(
            result_3d_df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color=color_param,
            hover_name=result_3d_df.index,
            hover_data=hover_data_list,
            labels=color_labels,
            title="UMAP 3D Embedding"
        )

        fig_3d.update_traces(marker=dict(size=5))
        fig_3d.update_layout(hovermode='closest')

        return {
            'umap_2d_plot': PlotlyResource(fig_2d),
            'umap_3d_plot': PlotlyResource(fig_3d),
            'umap_2d_table': Table(result_2d_df),
            'umap_3d_table': Table(result_3d_df)
        }
