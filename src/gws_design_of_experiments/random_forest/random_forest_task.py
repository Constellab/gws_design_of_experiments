from gws_core import (PlotlyResource, FloatParam, StrParam, ListParam, ConfigParams, ConfigSpecs, TypingStyle, InputSpec, InputSpecs, OutputSpec, OutputSpecs, Task, TaskInputs, TaskOutputs, task_decorator)
import numpy as np
import pandas as pd
from gws_core import Table

from gws_design_of_experiments.random_forest.random_forest_regressor import AutomaticRandomForestRegressor


@task_decorator(unique_name="RandomForestRegressorTask", human_name="Random Forest Regressor",
                style = TypingStyle.material_icon(material_icon_name="account_tree",
                background_color="#57970e"))


class RandomForestRegressorTask(Task):
    """
    Random Forest Regression Task with automatic hyperparameter tuning.
    
    This task performs Random Forest regression with cross-validation to find optimal
    hyperparameters (n_estimators and max_depth). It provides comprehensive outputs
    including metrics, feature importances, and visualization plots.
    
    Inputs:
        - data: Input table containing features and target variable
        
    Outputs:
        - summary_table: Performance metrics (R², RMSE) for train and test sets
        - vip_table: Feature importance scores ranked by importance with correlation signs
        - plot_estimators: Cross-validation performance vs hyperparameters
        - vip_plot: Bar plot of feature importances colored by correlation sign
        - plot_train_set: Predicted vs true values for training set
        - plot_test_set: Predicted vs true values for test set
        
    Configuration:
        - target: Name of the target column to predict
        - columns_to_exclude: List of column names to exclude from analysis (optional)
        - test_size: Proportion of data to use for testing (0.0 to 1.0)
    """

    input_specs: InputSpecs = InputSpecs({'data': InputSpec(Table, human_name='Input Data')})
    output_specs: OutputSpecs = OutputSpecs({'summary_table': OutputSpec(Table, human_name="Summary Table"),
                                          'vip_table': OutputSpec(Table, human_name="Variable Importance Table"),
                                          'plot_estimators': OutputSpec(PlotlyResource, human_name="Estimators Plot"),
                                          'vip_plot': OutputSpec(PlotlyResource, human_name="Variable Importance Plot"),
                                          'plot_train_set': OutputSpec(PlotlyResource, human_name="Train Predictions Plot"),
                                          'plot_test_set': OutputSpec(PlotlyResource, human_name="Test Predictions Plot")})
    config_specs: ConfigSpecs = ConfigSpecs({
                'target': StrParam(
                human_name="Target Column"),
                'columns_to_exclude': ListParam(
                human_name="Columns to Exclude",
                short_description="List of column names to exclude from RandomForest analysis",
                optional=True),
                'test_size': FloatParam(
                default_value=0.2,
                min_value=0.0,
                max_value=1.0,
                human_name="Test size",
                short_description="Proportion of the dataset to include in the test split (between 0.0 and 1.0).")})

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        """
        Execute the Random Forest regression task.
        
        Parameters
        ----------
        params : ConfigParams
            Configuration parameters including target, columns_to_exclude, and test_size
        inputs : TaskInputs
            Input data table
            
        Returns
        -------
        TaskOutputs
            Dictionary containing summary tables and plots
        """
        df = inputs['data'].get_data()
        X = df
        cols_to_drop = list(set((params['columns_to_exclude'] if params['columns_to_exclude'] else []) + [params['target']]))

        X = self.remove_constant_columns(X.drop(cols_to_drop, axis=1))
        y = df[params['target']]
        target_name = params['target']
        test_size = params['test_size']

        metrics_df, importance_df, fig_cv, fig_imp, fig_train, fig_test = self.main(X, y, target_name, test_size)

        return {"summary_table": metrics_df,
                "vip_table": importance_df,
                "plot_estimators": PlotlyResource(fig_cv),
                "vip_plot": PlotlyResource(fig_imp),
                "plot_train_set": PlotlyResource(fig_train),
                "plot_test_set": PlotlyResource(fig_test),
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

    def main(self, X, y, target_name, test_size):
        """
        Main execution logic for Random Forest regression.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        y : Series or array
            Target variable
        target_name : str
            Name of the target variable
        test_size : float
            Proportion of data for testing
            
        Returns
        -------
        tuple
            (metrics_df, importance_df, fig_cv, fig_imp, fig_train, fig_test)
        """
        rf_model = AutomaticRandomForestRegressor(
            test_size=test_size,
            n_splits=5,
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
            },
            random_state=42,
            target_name=target_name,
        )
        rf_model.fit(X, y)

        metrics_df = rf_model.metrics_df
        importance_df = rf_model.importance_df.head()

        fig_cv = rf_model.plot_cv_scores()
        fig_imp = rf_model.plot_feature_importances(top_n=10)
        fig_train = rf_model.plot_predictions(split="train")
        fig_test = rf_model.plot_predictions(split="test")

        return (metrics_df, importance_df, fig_cv, fig_imp, fig_train, fig_test)