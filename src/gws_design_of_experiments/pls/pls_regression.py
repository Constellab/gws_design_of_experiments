import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

class AutomaticPLSRegressor:
    def __init__(
        self,
        test_size=None,          # e.g. 0.2; if None -> no test set
        n_splits=5,
        max_components=None,     # if None -> min(n_samples-1, n_features)
        random_state=42,
        scale=True,
        target_name=None,
    ):
        self.test_size = test_size
        self.n_splits = n_splits
        self.max_components = max_components
        self.random_state = random_state
        self.scale = scale
        self._target_name_override = target_name

        # Will be filled after fit()
        self.model_ = None
        self.feature_names_ = None
        self.target_names_ = None
        self.cv_results_df_ = None
        self.metrics_df_ = None
        self.vip_df_ = None

        self.X_train_ = None
        self.y_train_ = None
        self.X_test_ = None
        self.y_test_ = None
        self.y_train_pred_ = None
        self.y_test_pred_ = None

    @staticmethod
    def _compute_vip(pls_model):
        """
        Compute VIP (Variable Importance in Projection) scores for a fitted
        PLSRegression model. Handles multi-output Y.

        VIP formula based on Wold’s definition:
        VIP_j = sqrt( p * sum_a(SSY_a * w_{j,a}^2 / sum_j w_{j,a}^2) / sum_a SSY_a )
        where SSY_a is variance in Y explained by component a.
        """
        T = pls_model.x_scores_      # (n_samples, n_components)
        W = pls_model.x_weights_     # (n_features, n_components)
        Q = pls_model.y_loadings_    # (n_targets, n_components)

        n_features = W.shape[0]

        # SSY_a for each component a:
        # SSY_a = (sum_i t_ia^2) * (sum_k q_ka^2)
        SSt = np.sum(T ** 2, axis=0)             # (n_components,)
        SSq = np.sum(Q ** 2, axis=0)             # (n_components,)
        SSY = SSt * SSq                          # (n_components,)
        total_SSY = np.sum(SSY)

        vip = np.zeros(n_features)
        for j in range(n_features):
            w_j = W[j, :]                        # (n_components,)
            w_norm_sq = np.sum(W ** 2, axis=0)   # (n_components,)
            weight_term = (w_j ** 2) / w_norm_sq
            vip[j] = np.sqrt(n_features * np.sum(SSY * weight_term) / total_SSY)

        return vip

    def fit(self, X, y):
        """
        Run automatic PLS regression with:
        - optional train/test split
        - CV to select best n_components
        - metrics computation (R2, RMSE) per output and per split
        - VIP computation
        """
        # Keep feature and target names if DataFrame
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(np.asarray(X).shape[1])]

        if hasattr(y, "columns"):
            self.target_names_ = list(y.columns)
        elif hasattr(y, "name") and y.ndim == 1:
            self.target_names_ = [y.name]
        else:
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                self.target_names_ = ["y"]
            else:
                self.target_names_ = [f"y{i}" for i in range(y_arr.shape[1])]

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Ensure y is 2D: (n_samples, n_targets)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape

        _max_components = min(n_samples - 1, n_features)
        if self.max_components is None:
            self.max_components = _max_components
        else:
            self.max_components = min(self.max_components, _max_components)

        # Optional train/test split
        if self.test_size is not None and self.test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_test = y_test = None

        self.X_train_, self.y_train_ = X_train, y_train
        self.X_test_, self.y_test_ = X_test, y_test

        # Hyperparameter tuning: number of components via CV (neg MSE)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        component_range = range(1, self.max_components + 1)
        cv_means = []
        cv_stds = []

        for n_comp in component_range:
            pls = PLSRegression(n_components=n_comp, scale=self.scale)
            scores = cross_val_score(
                pls,
                X_train,
                y_train,
                cv=kf,
                scoring="neg_mean_squared_error",  # averaged over outputs
            )
            cv_means.append(scores.mean())
            cv_stds.append(scores.std())

        cv_means = np.array(cv_means)
        cv_stds = np.array(cv_stds)

        # Store CV results as DataFrame
        cv_df = pd.DataFrame({
            "n_components": list(component_range),
            "cv_neg_mse_mean": cv_means,
            "cv_neg_mse_std": cv_stds,
        })
        cv_df["cv_rmse_mean"] = np.sqrt(-cv_df["cv_neg_mse_mean"])
        self.cv_results_df_ = cv_df

        # Best number of components
        best_idx = int(np.argmax(cv_means))
        best_n_components = component_range[best_idx]

        # Fit final model
        best_pls = PLSRegression(n_components=best_n_components, scale=self.scale)
        best_pls.fit(X_train, y_train)
        self.model_ = best_pls

        # Predictions & metrics (per target)
        y_train_pred = best_pls.predict(X_train)  # (n_samples, n_targets)
        self.y_train_pred_ = y_train_pred

        train_r2 = r2_score(y_train, y_train_pred, multioutput="raw_values")
        train_rmse = np.sqrt(
            mean_squared_error(y_train, y_train_pred, multioutput="raw_values")
        )

        rows = []
        for i, tname in enumerate(self.target_names_):
            rows.append({
                "split": "train",
                "target": tname,
                "best_n_components": best_n_components,
                "R2": train_r2[i],
                "RMSE": train_rmse[i],
            })

        y_test_pred = None
        if X_test is not None:
            y_test_pred = best_pls.predict(X_test)
            self.y_test_pred_ = y_test_pred

            test_r2 = r2_score(y_test, y_test_pred, multioutput="raw_values")
            test_rmse = np.sqrt(
                mean_squared_error(y_test, y_test_pred, multioutput="raw_values")
            )

            for i, tname in enumerate(self.target_names_):
                rows.append({
                    "split": "test",
                    "target": tname,
                    "best_n_components": best_n_components,
                    "R2": test_r2[i],
                    "RMSE": test_rmse[i],
                })
        else:
            self.y_test_pred_ = None

        self.metrics_df_ = pd.DataFrame(rows)

        # VIP scores
        vip_values = self._compute_vip(best_pls)

        # Extract regression coefficients
        # For multi-output, coef_ has shape (n_targets, n_features)
        # For single output, coef_ has shape (1, n_features)
        coefs = best_pls.coef_  # (1, n_features) or (n_targets, n_features)

        # Always squeeze to handle both cases, then average if needed
        if coefs.shape[0] == 1:
            # Single output: shape (1, n_features) -> (n_features,)
            avg_coefs = coefs.squeeze()
        else:
            # Multi-output: average across targets (axis=0)
            # TODO see to display one graph per target instead
            avg_coefs = np.mean(coefs, axis=0)

        self.vip_df_ = pd.DataFrame({
            "feature": self.feature_names_,
            "VIP": vip_values,
            "coefficient": avg_coefs,
            "sign": ["+" if c >= 0 else "-" for c in avg_coefs]
        }).sort_values("VIP", ascending=False).reset_index(drop=True)

        return self

    # ---- Accessors ----
    @property
    def metrics_df(self):
        return self.metrics_df_

    @property
    def vip_df(self):
        return self.vip_df_

    @property
    def cv_results_df(self):
        return self.cv_results_df_

    # ---- Plotting methods (Plotly) ----
    def plot_vip(self, top_n=None):
        """
        Bar plot of VIP scores with color coding based on coefficient sign.
        If top_n is provided, only show the top N features.
        Features are displayed in order of VIP importance (mixed +/- coefficients).
        """
        if self.vip_df_ is None:
            raise RuntimeError("Model not fitted yet. Call .fit(X, y) first.")

        df = self.vip_df_.copy()
        if top_n is not None:
            df = df.head(top_n)

        # Ensure features display in VIP order (already sorted in vip_df_)
        fig = px.bar(
            df,
            x="feature",
            y="VIP",
            color="sign",
            title="PLS VIP scores (colored by regresssion coefficient sign)",
            color_discrete_map={"+": "steelblue", "-": "coral"},
            hover_data={"coefficient": ":.4f", "VIP": ":.4f", "sign": True},
            category_orders={"feature": df["feature"].tolist()},  # Preserve VIP order
            text=df["coefficient"].round(2)  # Add coefficient values as text
        )
        fig.update_traces(textposition='outside')  # Place text above bars
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="VIP score",
            legend_title="Regression coefficient sign"
        )
        return fig

    def plot_cv_scores(self, use_rmse=True):
        """
        Plot cross-validation performance as a function of number of components.
        - use_rmse=True: plot RMSE
        - use_rmse=False: plot negative MSE
        """
        if self.cv_results_df_ is None:
            raise RuntimeError("Model not fitted yet. Call .fit(X, y) first.")

        df = self.cv_results_df_.copy()

        if use_rmse:
            y_col = "cv_rmse_mean"
            y_label = "CV RMSE (mean)"
            title = "Cross-validation RMSE vs. number of components"
        else:
            y_col = "cv_neg_mse_mean"
            y_label = "CV negative MSE (mean)"
            title = "Cross-validation negative MSE vs. number of components"

        fig = px.line(
            df,
            x="n_components",
            y=y_col,
            error_y="cv_neg_mse_std" if not use_rmse else None,
            markers=True,
            title=title,
        )
        fig.update_layout(
            xaxis_title="Number of components",
            yaxis_title=y_label
        )

        # Mark best n_components
        if self.model_ is not None:
            best_n = self.model_.n_components
            best_val = df.loc[df["n_components"] == best_n, y_col].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[best_n],
                    y=[best_val],
                    mode="markers+text",
                    text=["Best"],
                    textposition="top center",
                    name="Best n_components",
                )
            )

        return fig

    def plot_predictions(self, split="train", add_identity_line=True):
        """
        Scatter plot of predicted vs real data.
        Handles multi-output by using facets per target.
        split : "train" or "test"
        add_identity_line : if True, add y = x reference line.
        """
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        if split == "train":
            if self.y_train_pred_ is None:
                raise RuntimeError("Model not fitted yet. Call .fit(X, y) first.")
            y_true = self.y_train_
            y_pred = self.y_train_pred_
        else:
            if self.y_test_ is None or self.y_test_pred_ is None:
                raise RuntimeError("No test set was used / predictions not available.")
            y_true = self.y_test_
            y_pred = self.y_test_pred_

        n_targets = y_true.shape[1]

        # Build long-format DataFrame for facetting
        rows = []
        for t_idx in range(n_targets):
            tname = self.target_names_[t_idx]
            for yt, yp in zip(y_true[:, t_idx], y_pred[:, t_idx]):
                rows.append({
                    "y_true": yt,
                    "y_pred": yp,
                    "target": tname,
                })

        df = pd.DataFrame(rows)

        fig = px.scatter(
            df,
            x="y_true",
            y="y_pred",
            facet_col="target" if n_targets > 1 else None,
            facet_col_wrap=2 if n_targets > 2 else None,
            title=f"Predicted vs. true values ({split} set)",
        )
        fig.update_layout(
            xaxis_title="True values",
            yaxis_title="Predicted values",
        )

        if add_identity_line:
            min_val = min(df["y_true"].min(), df["y_pred"].min())
            max_val = max(df["y_true"].max(), df["y_pred"].max())

            # Add identity line to each subplot/facet
            for t_idx in range(n_targets):
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="y = x" if t_idx == 0 else None,
                        showlegend=(t_idx == 0),
                        line=dict(dash="dash", color="gray"),
                    ),
                    row=1 if n_targets <= 2 else (t_idx // 2 + 1),
                    col=(t_idx % 2 + 1) if n_targets > 1 else 1,
                )

        return fig