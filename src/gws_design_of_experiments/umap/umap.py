import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gws_core import (
    BoolParam,
    ConfigParams,
    ConfigSpecs,
    FloatParam,
    InputSpec,
    InputSpecs,
    IntParam,
    ListParam,
    OutputSpec,
    OutputSpecs,
    PlotlyResource,
    StrParam,
    Table,
    Task,
    TaskInputs,
    TaskOutputs,
    TypingStyle,
    task_decorator,
)
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from umap import UMAP


@task_decorator(
    "UMAPTask",
    human_name="UMAP Dimensionality Reduction",
    short_description="UMAP for dimensionality reduction and visualization",
    style=TypingStyle.material_icon(material_icon_name="scatter_plot", background_color="#85c5fe"),
)
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

    METRICS_OPTION = [
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "canberra",
        "braycurtis",
        "mahalanobis",
        "wminkowski",
        "seuclidean",
        "cosine",
        "correlation",
        "haversine",
        "hamming",
        "jaccard",
        "dice",
        "russelrao",
        "kulsinski",
        "ll_dirichlet",
        "hellinger",
        "rogerstanimoto",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ]
    input_specs = InputSpecs(
        {"data": InputSpec(Table, human_name="Data", short_description="Input data for UMAP")}
    )

    config_specs = ConfigSpecs(
        {
            "n_neighbors": IntParam(
                default_value=15,
                min_value=2,
                max_value=200,
                human_name="Number of Neighbors",
                short_description="Controls how UMAP balances local vs global structure",
            ),
            "min_dist": FloatParam(
                default_value=0.1,
                min_value=0.0,
                max_value=0.99,
                human_name="Minimum Distance",
                short_description="Minimum distance between points in the embedding",
            ),
            "metric": StrParam(
                default_value="euclidean",
                allowed_values=METRICS_OPTION,
                human_name="Distance Metric",
                short_description="Distance metric to use (euclidean, manhattan, cosine, etc.)",
            ),
            "scale_data": BoolParam(
                default_value=True,
                human_name="Scale Data",
                short_description="Whether to scale the data before applying UMAP",
            ),
            "n_clusters": IntParam(
                min_value=2,
                human_name="Number of Clusters",
                short_description="Number of clusters for K-Means clustering (optional)",
                optional=True,
            ),
            "color_by": ListParam(
                human_name="Color By Columns",
                short_description="Column names to use for colouring the points (optional). By default, all columns are used.",
                optional=True,
            ),
            "columns_to_exclude": ListParam(
                human_name="Columns to Exclude",
                short_description="List of column names to exclude from UMAP analysis",
                optional=True,
            ),
            "hover_data_columns": ListParam(
                human_name="Hover Data Columns",
                short_description="List of column names to display as metadata on hover",
                optional=True,
            ),
        }
    )

    output_specs = OutputSpecs(
        {
            "umap_2d_plot": OutputSpec(
                PlotlyResource,
                human_name="UMAP 2D Plot",
                short_description="Interactive UMAP 2D embedding visualization",
            ),
            "umap_3d_plot": OutputSpec(
                PlotlyResource,
                human_name="UMAP 3D Plot",
                short_description="Interactive UMAP 3D embedding visualization",
            ),
            "umap_2d_table": OutputSpec(
                Table,
                human_name="UMAP 2D Table",
                short_description="Table with UMAP 2D coordinates and cluster assignments",
            ),
            "umap_3d_table": OutputSpec(
                Table,
                human_name="UMAP 3D Table",
                short_description="Table with UMAP 3D coordinates and cluster assignments",
            ),
        }
    )

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        # Load data
        df = inputs["data"].get_data()

        # Extract hover data columns if specified
        hover_data = {}
        if params["hover_data_columns"]:
            hover_cols = [col.strip() for col in params["hover_data_columns"]]
            # Validate that columns exist
            invalid_cols = [col for col in hover_cols if col not in df.columns]
            if invalid_cols:
                raise ValueError(f"Hover data columns not found in data: {', '.join(invalid_cols)}")

            for col in hover_cols:
                hover_data[col] = df[col].copy()

        # Preserve color column BEFORE excluding columns
        color_column = None
        color_column_is_numeric = False
        if params["color_by"]:
            list_columns_to_color_by = params["color_by"].copy()
            for col in list_columns_to_color_by:
                if col not in df.columns:
                    list_columns_to_color_by = list_columns_to_color_by.remove(col)
                    raise ValueError(f"Color by column '{col}' not found in data.")
        else:
            list_columns_to_color_by = df.columns.tolist()
        df_color = df.copy()

        # Exclude specified columns
        columns_to_drop = []
        if params["columns_to_exclude"]:
            columns_to_drop = list(params["columns_to_exclude"])
            # Validate that columns exist
            invalid_cols = [col for col in columns_to_drop if col not in df.columns]
            if invalid_cols:
                raise ValueError(f"Columns not found in data: {', '.join(invalid_cols)}")

        # Remove duplicates
        columns_to_drop = list(set(columns_to_drop))

        # Drop excluded columns
        x = df.drop(columns=columns_to_drop).copy() if columns_to_drop else df.copy()

        # Check for categorical variables
        if x.select_dtypes(include=["object", "category", "string"]).shape[1] > 0:
            raise ValueError(
                "Categorical variables are not supported. Please encode or remove them or use them to color the points."
            )

        # Store original index
        original_index = x.index

        # Scaling if requested
        if params["scale_data"]:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
        else:
            x_scaled = x.values

        # Apply UMAP for 2D
        reducer_2d = UMAP(
            n_components=2,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            random_state=42,
            metric=params["metric"],
        )
        embedding_2d = reducer_2d.fit_transform(x_scaled)

        # Apply UMAP for 3D
        reducer_3d = UMAP(
            n_components=3,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            random_state=42,
            metric=params["metric"],
        )
        embedding_3d = reducer_3d.fit_transform(x_scaled)

        embedding_2d = np.asarray(
            embedding_2d[0] if isinstance(embedding_2d, tuple) else embedding_2d
        )
        embedding_3d = np.asarray(
            embedding_3d[0] if isinstance(embedding_3d, tuple) else embedding_3d
        )

        # Create 2D result dataframe
        result_2d_df = pd.DataFrame(
            {"UMAP1": embedding_2d[:, 0], "UMAP2": embedding_2d[:, 1]}, index=original_index
        )

        # Create 3D result dataframe
        result_3d_df = pd.DataFrame(
            {"UMAP1": embedding_3d[:, 0], "UMAP2": embedding_3d[:, 1], "UMAP3": embedding_3d[:, 2]},
            index=original_index,
        )

        # Perform clustering if requested (on 2D embedding)
        cluster_labels = None
        if params["n_clusters"] is not None:
            kmeans = KMeans(n_clusters=params["n_clusters"], random_state=42)
            cluster_labels = kmeans.fit_predict(embedding_2d)
            result_2d_df["Cluster"] = cluster_labels.astype(str)
            result_3d_df["Cluster"] = cluster_labels.astype(str)

        # Add all color-by columns to result dataframes
        for col in list_columns_to_color_by:
            color_column = df_color[col].copy()
            # Check if color column is numeric
            color_column_is_numeric = pd.api.types.is_numeric_dtype(color_column)

            if color_column_is_numeric:
                # Calculate skewness to determine if log transform is beneficial
                skewness = stats.skew(color_column.dropna())
                # Common threshold: skewness > 2 indicates high positive skew
                if skewness > 2:
                    # Add small constant to avoid log(0)
                    min_positive = (
                        color_column[color_column > 0].min() if (color_column > 0).any() else 1
                    )
                    color_column_transformed = np.log10(color_column + min_positive * 0.01)
                    # Store both original and transformed
                    result_2d_df[col] = df_color[col].values
                    result_3d_df[col] = df_color[col].values
                    result_2d_df[f"{col} (log10)"] = color_column_transformed.values
                    result_3d_df[f"{col} (log10)"] = color_column_transformed.values
                else:
                    result_2d_df[col] = color_column.values
                    result_3d_df[col] = color_column.values
            else:
                result_2d_df[col] = color_column.values.astype(str)
                result_3d_df[col] = color_column.values.astype(str)

        # Add hover data columns
        for col_name, col_data in hover_data.items():
            result_2d_df[col_name] = col_data.values
            result_3d_df[col_name] = col_data.values

        # Prepare hover data list for plots
        hover_data_list = list(hover_data.keys()) if hover_data else []

        # Create traces and dropdown menus for color options
        traces_2d = []
        traces_3d = []
        trace_groups_2d = {}
        trace_groups_3d = {}

        # Add cluster coloring option if clustering was performed
        if params["n_clusters"] is not None:
            cluster_traces_2d = []
            cluster_traces_3d = []

            for cluster in result_2d_df["Cluster"].unique():
                cluster_mask = result_2d_df["Cluster"] == cluster

                # 2D trace
                hover_text = result_2d_df.index[cluster_mask].tolist()
                hover_data_dict = {col: result_2d_df[col][cluster_mask] for col in hover_data_list}

                # Build hover template with formatting
                hover_template = "<b>%{hovertext}</b><br>"
                if hover_data_list:
                    for i, col in enumerate(hover_data_list):
                        # Check if column is numeric
                        if pd.api.types.is_numeric_dtype(result_2d_df[col]):
                            hover_template += f"{col}: %{{customdata[{i}]:.2f}}<br>"
                        else:
                            hover_template += f"{col}: %{{customdata[{i}]}}<br>"
                    hover_template = hover_template.rstrip("<br>") + "<extra></extra>"
                else:
                    hover_template = "<b>%{hovertext}</b><extra></extra>"

                cluster_traces_2d.append(
                    go.Scatter(
                        x=result_2d_df.loc[cluster_mask, "UMAP1"],
                        y=result_2d_df.loc[cluster_mask, "UMAP2"],
                        mode="markers",
                        marker={"size": 8},
                        name=f"Cluster {cluster}",
                        hovertext=hover_text,
                        customdata=np.column_stack(
                            [hover_data_dict[col] for col in hover_data_list]
                        )
                        if hover_data_list
                        else None,
                        hovertemplate=hover_template,
                        visible=False,
                    )
                )

                # 3D trace with same hover template logic
                hover_template_3d = "<b>%{hovertext}</b><br>"
                if hover_data_list:
                    for i, col in enumerate(hover_data_list):
                        if pd.api.types.is_numeric_dtype(result_3d_df[col]):
                            hover_template_3d += f"{col}: %{{customdata[{i}]:.2f}}<br>"
                        else:
                            hover_template_3d += f"{col}: %{{customdata[{i}]}}<br>"
                    hover_template_3d = hover_template_3d.rstrip("<br>") + "<extra></extra>"
                else:
                    hover_template_3d = "<b>%{hovertext}</b><extra></extra>"

                cluster_traces_3d.append(
                    go.Scatter3d(
                        x=result_3d_df.loc[cluster_mask, "UMAP1"],
                        y=result_3d_df.loc[cluster_mask, "UMAP2"],
                        z=result_3d_df.loc[cluster_mask, "UMAP3"],
                        mode="markers",
                        marker={"size": 5},
                        name=f"Cluster {cluster}",
                        hovertext=hover_text,
                        customdata=np.column_stack(
                            [hover_data_dict[col] for col in hover_data_list]
                        )
                        if hover_data_list
                        else None,
                        hovertemplate=hover_template_3d,
                        visible=False,
                    )
                )

            trace_groups_2d["Cluster"] = cluster_traces_2d
            trace_groups_3d["Cluster"] = cluster_traces_3d
            traces_2d.extend(cluster_traces_2d)
            traces_3d.extend(cluster_traces_3d)

        # Add traces for each column in list_columns_to_color_by
        for col in list_columns_to_color_by:
            color_data = df_color[col].copy()
            is_numeric = pd.api.types.is_numeric_dtype(color_data)

            if is_numeric:
                # Check if log transformation is beneficial
                skewness = stats.skew(color_data.dropna())
                apply_log = skewness > 2

                if apply_log:
                    min_positive = color_data[color_data > 0].min() if (color_data > 0).any() else 1
                    color_data_transformed = np.log10(color_data + min_positive * 0.01)
                    col_label = f"{col} (log10)"
                else:
                    color_data_transformed = color_data
                    col_label = col

                # Continuous coloring - single trace
                hover_text = result_2d_df.index.tolist()

                # Filter out hover data columns that match the color column
                filtered_hover_list = [
                    hcol for hcol in hover_data_list if hcol not in (col, col_label)
                ]
                hover_data_dict = {hcol: result_2d_df[hcol] for hcol in filtered_hover_list}

                # Build hover template with formatting
                hover_template = f"<b>%{{hovertext}}</b><br>{col_label}: %{{marker.color:.2f}}<br>"
                if filtered_hover_list:
                    for i, hcol in enumerate(filtered_hover_list):
                        if pd.api.types.is_numeric_dtype(result_2d_df[hcol]):
                            hover_template += f"{hcol}: %{{customdata[{i}]:.2f}}<br>"
                        else:
                            hover_template += f"{hcol}: %{{customdata[{i}]}}<br>"
                    hover_template = hover_template.rstrip("<br>") + "<extra></extra>"
                else:
                    hover_template = f"<b>%{{hovertext}}</b><br>{col_label}: %{{marker.color:.2f}}<extra></extra>"

                trace_2d = go.Scatter(
                    x=result_2d_df["UMAP1"],
                    y=result_2d_df["UMAP2"],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": color_data_transformed,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": col_label},
                    },
                    name=col_label,
                    hovertext=hover_text,
                    customdata=np.column_stack(
                        [hover_data_dict[hcol] for hcol in filtered_hover_list]
                    )
                    if filtered_hover_list
                    else None,
                    hovertemplate=hover_template,
                    visible=False,
                )

                # 3D trace with same hover template logic
                hover_data_dict_3d = {hcol: result_3d_df[hcol] for hcol in filtered_hover_list}

                hover_template_3d = (
                    f"<b>%{{hovertext}}</b><br>{col_label}: %{{marker.color:.2f}}<br>"
                )
                if filtered_hover_list:
                    for i, hcol in enumerate(filtered_hover_list):
                        if pd.api.types.is_numeric_dtype(result_3d_df[hcol]):
                            hover_template_3d += f"{hcol}: %{{customdata[{i}]:.2f}}<br>"
                        else:
                            hover_template_3d += f"{hcol}: %{{customdata[{i}]}}<br>"
                    hover_template_3d = hover_template_3d.rstrip("<br>") + "<extra></extra>"
                else:
                    hover_template_3d = f"<b>%{{hovertext}}</b><br>{col_label}: %{{marker.color:.2f}}<extra></extra>"

                trace_3d = go.Scatter3d(
                    x=result_3d_df["UMAP1"],
                    y=result_3d_df["UMAP2"],
                    z=result_3d_df["UMAP3"],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": color_data_transformed,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": col_label},
                    },
                    name=col_label,
                    hovertext=hover_text,
                    customdata=np.column_stack(
                        [hover_data_dict_3d[hcol] for hcol in filtered_hover_list]
                    )
                    if filtered_hover_list
                    else None,
                    hovertemplate=hover_template_3d,
                    visible=False,
                )

                trace_groups_2d[col_label] = [trace_2d]
                trace_groups_3d[col_label] = [trace_3d]
                traces_2d.append(trace_2d)
                traces_3d.append(trace_3d)
            else:
                # Categorical coloring - multiple traces
                cat_traces_2d = []
                cat_traces_3d = []

                # Get unique categories and assign colors
                unique_categories = color_data.unique()
                colors = px.colors.qualitative.Plotly

                for idx, category in enumerate(unique_categories):
                    cat_mask = color_data == category
                    hover_text = result_2d_df.index[cat_mask].tolist()

                    # Filter out hover data columns that match the color column
                    filtered_hover_list = [hcol for hcol in hover_data_list if hcol != col]
                    hover_data_dict = {
                        hcol: result_2d_df[hcol][cat_mask] for hcol in filtered_hover_list
                    }

                    color = colors[idx % len(colors)]

                    # Build hover template with formatting
                    hover_template = f"<b>%{{hovertext}}</b><br>{col}: {category}<br>"
                    if filtered_hover_list:
                        for i, hcol in enumerate(filtered_hover_list):
                            if pd.api.types.is_numeric_dtype(result_2d_df[hcol]):
                                hover_template += f"{hcol}: %{{customdata[{i}]:.2f}}<br>"
                            else:
                                hover_template += f"{hcol}: %{{customdata[{i}]}}<br>"
                        hover_template = hover_template.rstrip("<br>") + "<extra></extra>"
                    else:
                        hover_template = (
                            f"<b>%{{hovertext}}</b><br>{col}: {category}<extra></extra>"
                        )

                    cat_traces_2d.append(
                        go.Scatter(
                            x=result_2d_df.loc[cat_mask, "UMAP1"],
                            y=result_2d_df.loc[cat_mask, "UMAP2"],
                            mode="markers",
                            marker={"size": 8, "color": color},
                            name=str(category),
                            hovertext=hover_text,
                            customdata=np.column_stack(
                                [hover_data_dict[hcol] for hcol in filtered_hover_list]
                            )
                            if filtered_hover_list
                            else None,
                            hovertemplate=hover_template,
                            visible=False,
                        )
                    )

                    # 3D trace with same hover template logic
                    hover_data_dict_3d = {
                        hcol: result_3d_df[hcol][cat_mask] for hcol in filtered_hover_list
                    }

                    hover_template_3d = f"<b>%{{hovertext}}</b><br>{col}: {category}<br>"
                    if filtered_hover_list:
                        for i, hcol in enumerate(filtered_hover_list):
                            if pd.api.types.is_numeric_dtype(result_3d_df[hcol]):
                                hover_template_3d += f"{hcol}: %{{customdata[{i}]:.2f}}<br>"
                            else:
                                hover_template_3d += f"{hcol}: %{{customdata[{i}]}}<br>"
                        hover_template_3d = hover_template_3d.rstrip("<br>") + "<extra></extra>"
                    else:
                        hover_template_3d = (
                            f"<b>%{{hovertext}}</b><br>{col}: {category}<extra></extra>"
                        )

                    cat_traces_3d.append(
                        go.Scatter3d(
                            x=result_3d_df.loc[cat_mask, "UMAP1"],
                            y=result_3d_df.loc[cat_mask, "UMAP2"],
                            z=result_3d_df.loc[cat_mask, "UMAP3"],
                            mode="markers",
                            marker={"size": 5, "color": color},
                            name=str(category),
                            hovertext=hover_text,
                            customdata=np.column_stack(
                                [hover_data_dict_3d[hcol] for hcol in filtered_hover_list]
                            )
                            if filtered_hover_list
                            else None,
                            hovertemplate=hover_template_3d,
                            visible=False,
                        )
                    )

                trace_groups_2d[col] = cat_traces_2d
                trace_groups_3d[col] = cat_traces_3d
                traces_2d.extend(cat_traces_2d)
                traces_3d.extend(cat_traces_3d)

        # Create dropdown buttons for 2D
        buttons_2d = []
        for label, group in trace_groups_2d.items():
            visibility = [False] * len(traces_2d)
            for t in group:
                visibility[traces_2d.index(t)] = True

            buttons_2d.append(
                {
                    "label": label,
                    "method": "update",
                    "args": [
                        {"visible": visibility},
                        {"title": f"UMAP 2D Embedding – Color by {label}"},
                    ],
                }
            )

        # Create dropdown buttons for 3D
        buttons_3d = []
        for label, group in trace_groups_3d.items():
            visibility = [False] * len(traces_3d)
            for t in group:
                visibility[traces_3d.index(t)] = True

            buttons_3d.append(
                {
                    "label": label,
                    "method": "update",
                    "args": [
                        {"visible": visibility},
                        {"title": f"UMAP 3D Embedding – Color by {label}"},
                    ],
                }
            )

        # Set first trace group as visible for 2D
        if trace_groups_2d:
            first_group_2d = list(trace_groups_2d.values())[0]
            for trace in first_group_2d:
                trace.visible = True

        # Set first trace group as visible for 3D
        if trace_groups_3d:
            first_group_3d = list(trace_groups_3d.values())[0]
            for trace in first_group_3d:
                trace.visible = True

        # Create 2D figure
        fig_2d = go.Figure(traces_2d)

        first_label_2d = list(trace_groups_2d.keys())[0] if trace_groups_2d else "UMAP"
        fig_2d.update_layout(
            title=f"UMAP 2D Embedding – Color by {first_label_2d}",
            updatemenus=[
                {
                    "buttons": buttons_2d,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.02,
                    "y": 1,
                }
            ]
            if len(buttons_2d) > 1
            else [],
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode="closest",
        )

        # Create 3D figure
        fig_3d = go.Figure(traces_3d)

        first_label_3d = list(trace_groups_3d.keys())[0] if trace_groups_3d else "UMAP"
        fig_3d.update_layout(
            title=f"UMAP 3D Embedding – Color by {first_label_3d}",
            updatemenus=[
                {
                    "buttons": buttons_3d,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.02,
                    "y": 1,
                }
            ]
            if len(buttons_3d) > 1
            else [],
            scene={
                "xaxis_title": "UMAP 1",
                "yaxis_title": "UMAP 2",
                "zaxis_title": "UMAP 3",
            },
            hovermode="closest",
        )

        return {
            "umap_2d_plot": PlotlyResource(fig_2d),
            "umap_3d_plot": PlotlyResource(fig_3d),
            "umap_2d_table": Table(result_2d_df),
            "umap_3d_table": Table(result_3d_df),
        }
