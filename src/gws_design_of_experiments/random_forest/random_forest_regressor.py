import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


class AutomaticRandomForestRegressor:
    """
    Automatic Random Forest regressor with cross-validation.

    - optional train/test split
    - CV over a grid of hyperparameters:
        * n_estimators
        * max_depth
    - metrics (R2, RMSE) in a DataFrame
    - feature importances in a DataFrame
    - Plotly visualizations:
        * CV curves vs hyperparameters
        * feature importance bar plot
        * predicted vs true scatter with R2 & RMSE in title
    """

    def __init__(
        self,
        test_size=None,  # e.g. 0.2; if None -> no test set
        n_splits=5,
        param_grid=None,  # dict with lists: {"n_estimators": [...], "max_depth": [...]}
        random_state=42,
        n_jobs=-1,  # for RandomForestRegressor
        target_name=None,  # explicit target name
    ):
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._target_name_override = target_name

        # Default hyperparameter grid
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 5, 10, 20],
            }
        self.param_grid = param_grid

        # Filled after fit()
        self.model_ = None
        self.feature_names_ = None
        self.target_name_ = None

        self.cv_results_df_ = None
        self.metrics_df_ = None
        self.importance_df_ = None

        self.x_train_ = None
        self.y_train_ = None
        self.x_test_ = None
        self.y_test_ = None

        self.y_train_pred_ = None
        self.y_test_pred_ = None

        self.best_params_ = None

    # -------------------- FIT -------------------- #

    def fit(self, x, y):
        """
        Fit RandomForestRegressor with cross-validation over the param grid
        and compute metrics and feature importances.

        Assumes 1D target y (single-output regression).
        """
        # Feature names
        if hasattr(x, "columns"):
            self.feature_names_ = list(x.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(np.asarray(x).shape[1])]

        x = np.asarray(x, dtype=float)

        # Target name - use override if provided, otherwise try to get from y
        if self._target_name_override is not None:
            self.target_name_ = self._target_name_override
        elif hasattr(y, "name") and y.name is not None:
            self.target_name_ = y.name
        else:
            self.target_name_ = "y"

        y = np.asarray(y, dtype=float)

        if y.ndim > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                "AutomaticRandomForestRegressor currently supports only 1D targets."
            )

        y = y.ravel()

        # Optional train/test split
        if self.test_size is not None and self.test_size > 0:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        else:
            x_train, y_train = x, y
            x_test = y_test = None

        self.x_train_, self.y_train_ = x_train, y_train
        self.x_test_, self.y_test_ = x_test, y_test
        # ---------------- Cross-validation over param grid ---------------- #
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        n_estimators_list = self.param_grid.get("n_estimators", [100])
        max_depth_list = self.param_grid.get("max_depth", [None])

        rows = []

        for n_est in n_estimators_list:
            for max_d in max_depth_list:
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_d,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                scores = cross_val_score(
                    rf,
                    x_train,
                    y_train,
                    cv=kf,
                    scoring="neg_mean_squared_error",
                )
                mean_neg_mse = scores.mean()
                std_neg_mse = scores.std()
                mean_rmse = np.sqrt(-mean_neg_mse)
                std_rmse = np.sqrt(std_neg_mse) if std_neg_mse >= 0 else np.nan

                rows.append(
                    {
                        "n_estimators": n_est,
                        "max_depth": max_d if max_d is not None else np.nan,
                        "cv_neg_mse_mean": mean_neg_mse,
                        "cv_neg_mse_std": std_neg_mse,
                        "cv_rmse_mean": mean_rmse,
                        "cv_rmse_std": std_rmse,
                    }
                )

        self.cv_results_df_ = pd.DataFrame(rows)

        # Choose best params: max mean neg MSE (equivalent to min RMSE)
        best_idx = int(self.cv_results_df_["cv_neg_mse_mean"].idxmax())
        best_row = self.cv_results_df_.iloc[best_idx]

        best_n_estimators = int(best_row["n_estimators"])
        best_max_depth = None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"])

        self.best_params_ = {
            "n_estimators": best_n_estimators,
            "max_depth": best_max_depth,
        }

        # ---------------- Fit final model ---------------- #
        rf_best = RandomForestRegressor(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        rf_best.fit(x_train, y_train)
        self.model_ = rf_best

        # ---------------- Metrics ---------------- #
        y_train_pred = self.model_.predict(x_train)
        self.y_train_pred_ = y_train_pred

        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        rows_metrics = [
            {
                "split": "train",
                "target": self.target_name_,
                "n_estimators": best_n_estimators,
                "max_depth": best_max_depth,
                "R2": train_r2,
                "RMSE": train_rmse,
            }
        ]

        if x_test is not None and y_test is not None:
            y_test_pred = self.model_.predict(x_test)
            self.y_test_pred_ = y_test_pred

            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            rows_metrics.append(
                {
                    "split": "test",
                    "target": self.target_name_,
                    "n_estimators": best_n_estimators,
                    "max_depth": best_max_depth,
                    "R2": test_r2,
                    "RMSE": test_rmse,
                }
            )
        else:
            self.y_test_pred_ = None

        self.metrics_df_ = pd.DataFrame(rows_metrics)

        # ---------------- Feature importances ---------------- #
        importances = self.model_.feature_importances_

        # Compute correlation with predictions for directionality
        x_train_arr = self.x_train_
        y_hat = self.y_train_pred_

        importance_rows = []
        for j, fname in enumerate(self.feature_names_):
            xj = x_train_arr[:, j]
            # Handle constant columns / nan corr
            if np.std(xj) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(xj, y_hat)[0, 1]
                if np.isnan(corr):
                    corr = 0.0

            imp = importances[j]
            importance_rows.append(
                {
                    "feature": fname,
                    "importance": imp,
                    "correlation": corr,
                    "sign": "+" if corr >= 0 else "-",
                }
            )

        self.importance_df_ = (
            pd.DataFrame(importance_rows)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return self

    # -------------------- PROPERTIES -------------------- #

    @property
    def metrics_df(self):
        return self.metrics_df_

    @property
    def importance_df(self):
        return self.importance_df_

    @property
    def cv_results_df(self):
        return self.cv_results_df_

    # -------------------- PLOTS -------------------- #

    def plot_feature_importances(self, top_n=None):
        """
        Bar plot of feature importances with color coding based on correlation sign.
        If top_n is provided, only show the top N features.
        Features are displayed in order of importance (mixed +/- correlations).
        """
        if self.importance_df_ is None:
            raise RuntimeError("Model not fitted yet. Call .fit(x, y) first.")

        df = self.importance_df_.copy()
        if top_n is not None:
            df = df.head(top_n)

        # Ensure features display in importance order (already sorted in importance_df_)
        fig = px.bar(
            df,
            x="feature",
            y="importance",
            color="sign",
            title="Random Forest feature importances (colored by correlation sign)",
            color_discrete_map={"+": "steelblue", "-": "coral"},
            hover_data={"importance": ":.4f", "sign": True},
            category_orders={"feature": df["feature"].tolist()},  # Preserve importance order
        )
        fig.update_traces(textposition="outside")  # Place text above bars
        fig.update_layout(
            xaxis_title="Feature", yaxis_title="Importance", legend_title="Correlation sign"
        )
        return fig

    def plot_cv_scores(self, use_rmse=True):
        """
        Plot cross-validation performance as a function of n_estimators & max_depth.
        x-axis is n_estimators; color is max_depth.
        """
        if self.cv_results_df_ is None:
            raise RuntimeError("Model not fitted yet. Call .fit(x, y) first.")

        df = self.cv_results_df_.copy()

        if use_rmse:
            y_col = "cv_rmse_mean"
            y_err = "cv_rmse_std"
            y_label = "CV RMSE (mean)"
        else:
            y_col = "cv_neg_mse_mean"
            y_err = "cv_neg_mse_std"
            y_label = "CV negative MSE (mean)"

        # Replace NaN max_depth (meaning None) for display
        df["max_depth_display"] = df["max_depth"].fillna(-1).astype(int).astype(str)
        df.loc[df["max_depth_display"] == "-1", "max_depth_display"] = "None"

        fig = px.line(
            df,
            x="n_estimators",
            y=y_col,
            error_y=y_err,
            color="max_depth_display",
            markers=True,
            title=f"Random Forest CV performance vs n_estimators ({y_label})",
        )

        fig.update_layout(
            xaxis_title="n_estimators",
            yaxis_title=y_label,
            legend_title="max_depth",
        )

        # Mark best hyperparameters
        if self.best_params_ is not None and len(df) > 0:
            best_n = self.best_params_["n_estimators"]
            best_d = self.best_params_["max_depth"]
            best_d_display = "None" if best_d is None else str(int(best_d))

            mask = (df["n_estimators"] == best_n) & (df["max_depth_display"] == best_d_display)
            idx = np.where(mask.to_numpy())[0]
            if idx.size > 0:
                i0 = int(idx[0])
                best_x = df.iloc[i0]["n_estimators"]
                best_y = df.iloc[i0][y_col]

                fig.add_trace(
                    go.Scatter(
                        x=[best_x],
                        y=[best_y],
                        mode="markers+text",
                        text=["Best"],
                        textposition="top center",
                        name="Best hyperparameters",
                    )
                )

        return fig

    def plot_predictions(self, split="train", add_identity_line=True):
        """
        Scatter plot of predicted vs real data.

        Title includes R2 and RMSE for the chosen split.
        """
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        if split == "train":
            if self.y_train_pred_ is None or self.y_train_ is None:
                raise RuntimeError("Model not fitted yet. Call .fit(x, y) first.")
            y_true = self.y_train_
            y_pred = self.y_train_pred_
        else:
            if self.y_test_ is None or self.y_test_pred_ is None:
                raise RuntimeError("No test set was used / predictions not available.")
            y_true = self.y_test_
            y_pred = self.y_test_pred_

        if y_true is None or y_pred is None:
            raise RuntimeError("Cannot compute metrics: predictions or true values are None.")

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics_txt = f"R² = {r2:.3f}, RMSE = {rmse:.3f}"

        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

        fig = px.scatter(
            df,
            x="y_true",
            y="y_pred",
            title=(f"Predicted vs true ({split} set) — Random Forest | {metrics_txt}"),
        )
        fig.update_layout(
            xaxis_title="True values",
            yaxis_title="Predicted values",
        )

        if add_identity_line:
            min_val = min(df["y_true"].min(), df["y_pred"].min())
            max_val = max(df["y_true"].max(), df["y_pred"].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="y = x",
                    line={"dash": "dash"},
                )
            )

        return fig
