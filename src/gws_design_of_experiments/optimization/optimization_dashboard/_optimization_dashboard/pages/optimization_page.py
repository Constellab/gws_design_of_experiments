import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from gws_core.streamlit import dataframe_paginated


def render_first_page(path_output_dir : str):
    # -----------------------------------------------------------------------------
    # --- Load optimization result -------------------------------------------------
    # -----------------------------------------------------------------------------
    file_path_opt_res = os.path.join(path_output_dir, "generalized_solutions.csv")

    df = pd.read_csv(file_path_opt_res)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    best_solution, surface_explorer, feature_importance_matrix, observed_vs_predicted, data_explorer = st.tabs(
        [   "✅ Optimal Solution",
            "📊 3D Surface Explorer",
            "📈 Feature Importance Matrix",
            "🔍 Observed vs Predicted",
            "🗂️ Data Explorer"
        ]
    )

    # -----------------------------------------------------------------------------
    # --- Best Solution -------------------------------------------------------------
    # -----------------------------------------------------------------------------
    with best_solution:
        file_path_best = os.path.join(path_output_dir, "best_generalized_solution.csv")
        col_tab, col_plots = st.columns([1, 1])

        with col_tab:
            df_best = pd.read_csv(file_path_best)

            # Calculate statistics for parameters
            stats_data = []
            for param in df.columns:
                mean_val = df[param].mean()
                std_val = df[param].std()
                optimal_val = df_best[param].iloc[0]


                stats_data.append({
                    'Parameters': param,
                    'Mean Value': safe_round(mean_val),
                    'Standard Deviation': safe_round(std_val),
                    'Optimal Value': safe_round(optimal_val)
                })

            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        with col_plots:
            # Multiselect for choosing parameters to plot
            available_params = [row['Parameters'] for row in stats_data]
            selected_params = st.selectbox(
                "Select parameter for boxplot:",
                options=available_params, index = 0
            )

            # Create boxplot for selected parameter
            if selected_params and selected_params in df.columns:
                fig = px.box(df, y=selected_params, title=f"Distribution of {selected_params}")
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------------
    # --- 3D Surface Explorer ------------------------------------------------------
    # -----------------------------------------------------------------------------
    with surface_explorer:
        cols = st.columns(3)
        with cols[0]:
            x_col = st.selectbox("X Axis", numeric_cols, index=0)
        with cols[1]:
            y_col = st.selectbox("Y Axis", numeric_cols, index=1)
        with cols[2]:
            z_col = st.selectbox("Z Axis", numeric_cols, index=2)

        X = df[[x_col, y_col]].values
        Z = df[z_col].values

        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        model.fit(X, Z)

        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 30)
        y_range = np.linspace(df[y_col].min(), df[y_col].max(), 30)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        xy_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        z_pred = model.predict(xy_grid).reshape(x_grid.shape)

        fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_pred,
                colorscale="Viridis",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode="markers",
                marker=dict(size=4, color="red"),
            )
        )
        fig.update_layout(
            scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View data", expanded=False):
            st.dataframe(df[[x_col, y_col, z_col]])

    # -----------------------------------------------------------------------------
    # --- Observed vs Predicted ----------------------------------------------------
    # -----------------------------------------------------------------------------
    with observed_vs_predicted:

        pred_file = os.path.join(path_output_dir, "actual_vs_predicted.csv")
        if not os.path.exists(pred_file):
            st.warning(f"The {pred_file} file cannot be found. Please generate the predictions first.")
            return
        df_obs_pred = pd.read_csv(pred_file)
        target_cols = [col for col in df_obs_pred.columns if not col.endswith("_pred")]

        selected_target = st.selectbox("Select target variable:", target_cols)
        pred_col = selected_target + "_pred"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_obs_pred[selected_target],
                y=df_obs_pred[pred_col],
                mode="markers",
                name="Points",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_obs_pred[selected_target],
                y=df_obs_pred[selected_target],
                mode="lines",
                name="Ideal",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title=f"{selected_target}: Observed vs Predicted",
            xaxis_title="Observed",
            yaxis_title="Predicted",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


    # -----------------------------------------------------------------------------
    # --- Feature Importance Matrix -----------------------------------------------
    # -----------------------------------------------------------------------------
    with feature_importance_matrix:

        importance_file = os.path.join(path_output_dir, "feature_importance_matrix.csv")
        if not os.path.exists(importance_file):
            st.warning("No 'feature_importance_matrix.csv' file found.")
            return
        imp_df = pd.read_csv(importance_file, index_col=0)
        all_targets = imp_df.columns.tolist()

        selected_targets = st.multiselect(
            "Choose the targets to display:", all_targets, default=all_targets
        )
        separate = st.checkbox("View separately", value=False)

        if not selected_targets:
            st.warning("Select at least one target to display the graph.")
            return
        if separate:
            for target in selected_targets:
                st.write(f"### {target} - Importance of variables")
                sorted_df = imp_df[target].sort_values(ascending=False).reset_index()
                sorted_df.columns = ["Feature", "Importance"]
                fig = px.bar(sorted_df, x="Feature", y="Importance", title=target)
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("### Combined importance")
            fig = go.Figure()
            for target in selected_targets:
                fig.add_trace(
                    go.Bar(name=target, x=imp_df.index, y=imp_df[target])
                )
            fig.update_layout(
                barmode="group",
                xaxis_title="Feature",
                yaxis_title="Importance",
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)


    # -----------------------------------------------------------------------------
    # --- Data Explorer --------------------------------------------------
    # -----------------------------------------------------------------------------
    with data_explorer:

        # --- Sorting options ------------------------------------------------------
        cols = st.columns(2)
        with cols[0]:
            default_col = 'CV_percent' if 'CV_percent' in df.columns else df.columns[0]
            sort_col = st.selectbox("Sort by column:", df.columns.tolist(), index=df.columns.tolist().index(default_col))
        with cols[1]:
            order = st.selectbox("Order:", ("Descending", "Ascending"), index=0)
            if order == "Ascending":
                is_ascending = True
            else:
                is_ascending = False


        # --- Sort and display -----------------------------------------------------
        sorted_df = df.sort_values(by=sort_col, ascending=is_ascending)

        dataframe_paginated(
                sorted_df, paginate_rows=True, row_page_size_options=[25, 50, 100],
                paginate_columns=False, column_page_size_options=None, key="data_explorer_table", use_container_width=True)

        st.caption(
            f"Data are sorted by **{sort_col}** ({order} order)."
            )

# Handle rounding for different data types
def safe_round(val, decimals=4):
    if isinstance(val, (bool, np.bool_)):
        return val
    try:
        return round(val, decimals)
    except (TypeError, AttributeError):
        return val