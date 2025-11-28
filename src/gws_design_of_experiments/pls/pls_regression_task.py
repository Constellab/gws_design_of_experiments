from gws_core import (ConfigParams, BoolParam, FloatParam, PlotlyResource, ListParam, TypingStyle, ConfigSpecs, InputSpec, InputSpecs, OutputSpec, OutputSpecs, Resource, Task, TaskInputs, TaskOutputs, task_decorator)
import numpy as np
import pandas as pd
from gws_core import Table
from gws_design_of_experiments.pls.pls_regression import AutomaticPLSRegressor


@task_decorator("PLSRegression", human_name="PLS Regression", short_description="PLS Regression",
                style=TypingStyle.material_icon(material_icon_name="insights",
                background_color="#c5fe85"))
class PLSRegressorTask(Task):
    """
    Partial Least Squares (PLS) Regression Task with automatic component selection.

    This task performs PLS regression with cross-validation to determine the optimal
    number of components. It supports both single and multi-output regression and
    provides VIP (Variable Importance in Projection) scores for feature interpretation.

    Inputs:
        - data: Input table containing features and target variable(s)

    Outputs:
        - summary_table: Performance metrics (R², RMSE) per target for train and test sets
        - vip_table: VIP scores ranked by importance
        - plot_components: Cross-validation performance vs number of components
        - vip_plot: Bar plot of VIP scores
        - plot_train_set: Predicted vs true values for training set (faceted by target)
        - plot_test_set: Predicted vs true values for test set (faceted by target)

    Configuration:
        - target: List of target column name(s) to predict
        - columns_to_exclude: List of column names to exclude from analysis (optional)
        - scale_data: Whether to scale the data before fitting (default: True)
        - test_size: Proportion of data to use for testing (0.0 to 1.0)
    """

    input_specs: InputSpecs = InputSpecs({'data': InputSpec(Table, human_name='Input Data')})
    output_specs: OutputSpecs = OutputSpecs({'summary_table': OutputSpec(Table, human_name="Summary Table"),
                                          'vip_table': OutputSpec(Table, human_name="Variable Importance Table"),
                                          'plot_components': OutputSpec(PlotlyResource, human_name="Components Plot"),
                                          'vip_plot': OutputSpec(PlotlyResource, human_name="Variable Importance Plot"),
                                          'plot_train_set': OutputSpec(PlotlyResource, human_name="Train Predictions Plot"),
                                          'plot_test_set': OutputSpec(PlotlyResource, human_name="Test Predictions Plot")})
    config_specs : ConfigSpecs = ConfigSpecs({
                'target': ListParam(
                human_name="Target Column"),
                'columns_to_exclude': ListParam(
                human_name="Columns to Exclude",
                short_description="List of column names to exclude from RandomForest analysis",
                optional=True),
                'scale_data': BoolParam(
                default_value=True,
                human_name="Scale Data",
                short_description="Whether to scale the data before fitting the PLS model."),
                'test_size': FloatParam(
                default_value=0.2,
                min_value=0.0,
                max_value=1.0,
                human_name="Test size",
                short_description="Proportion of the dataset to include in the test split (between 0.0 and 1.0).")})

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        """
        Execute the PLS regression task.

        Parameters
        ----------
        params : ConfigParams
            Configuration parameters including target, columns_to_exclude,
            scale_data, and test_size
        inputs : TaskInputs
            Input data table

        Returns
        -------
        TaskOutputs
            Dictionary containing summary tables and plots
        """
        df = inputs['data'].get_data()
        X = df
        cols_to_drop = list(set((params['columns_to_exclude'] if params['columns_to_exclude'] else []) + params['target']))

        X = self.remove_constant_columns(X.drop(cols_to_drop, axis=1))
        y = df[params['target']]
        target_name = str(params['target'])
        test_size = params['test_size']
        scale_data = params['scale_data']

        metric_df, vip_df, fig_vip, fig_cv, fig_pred_train, fig_pred_test = self.main(X,y, target_name, test_size, scale_data)

        return {"summary_table": metric_df,
                "vip_table": vip_df,
                "plot_components": PlotlyResource(fig_cv),
                "vip_plot": PlotlyResource(fig_vip),
                "plot_train_set": PlotlyResource(fig_pred_train),
                "plot_test_set": PlotlyResource(fig_pred_test),
               }


    def remove_constant_columns(self, X):
        """
        Remove constant (zero-variance) columns from X.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Input feature matrix

        Returns
        -------
        X_clean : DataFrame or array
            Feature matrix with constant columns removed
        """
        # If pandas → extract names, else create generic names
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Compute variance for each column
        variances = np.var(X_arr, axis=0)

        # Find constant columns
        kept_idx = np.where(variances != 0)[0]

        # Produce cleaned output
        if isinstance(X, pd.DataFrame):
            X_clean = X.iloc[:, kept_idx]
        else:
            X_clean = X_arr[:, kept_idx]

        return X_clean

    def main(self, X, y, target_name, test_size, scale_data):
        """
        Main execution logic for PLS regression.

        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        y : DataFrame or Series
            Target variable(s)
        target_name : str
            Name(s) of the target variable(s)
        test_size : float
            Proportion of data for testing
        scale_data : bool
            Whether to scale the data

        Returns
        -------
        tuple
            (metric_df, vip_df, fig_vip, fig_cv, fig_pred_train, fig_pred_test)
        """
        model = AutomaticPLSRegressor(test_size=test_size, n_splits=5, max_components=10, target_name=target_name, scale=scale_data)
        model.fit(X, y)

        metric_df = model.metrics_df
        vip_df = model.vip_df

        fig_vip = model.plot_vip(top_n=20)
        fig_cv = model.plot_cv_scores(use_rmse=True)
        fig_pred_train = model.plot_predictions(split="train")
        fig_pred_test = model.plot_predictions(split="test")

        return (metric_df, vip_df, fig_vip, fig_cv, fig_pred_train, fig_pred_test)