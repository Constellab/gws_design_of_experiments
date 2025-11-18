from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import plotly.graph_objects as go


from gws_core import (ConfigParams, ConfigSpecs, InputSpec, InputSpecs,
                      OutputSpec, OutputSpecs, ParamSet,
                      Table, Task, TaskInputs, TaskOutputs, TypingStyle, PlotlyResource,
                      task_decorator, StrParam, FloatParam, IntParam, BoolParam)


@task_decorator("PLSRegression", human_name="PLS Regression", short_description="PLS Regression",
                style=TypingStyle.material_icon(material_icon_name="insights",
                                                background_color="#c5fe85"))
class PLS(Task):
    """
    Performs Partial Least Squares (PLS) Regression analysis on tabular data.

    This task implements PLS regression with automatic component selection,
    data scaling options, and comprehensive visualization of results including
    Variable Importance in Projection (VIP) scores.

    The task performs the following steps:
    1. Splits data into training and test sets
    2. Optionally scales the data using StandardScaler
    3. Determines optimal number of PLS components (or uses specified value)
    4. Fits PLS model and evaluates performance
    5. Calculates VIP scores for feature importance
    6. Generates visualization plots

    Inputs:
        - data: Table containing features and target variable(s)

    Outputs:
        - pls_component_plot: Plot showing MSE vs number of components
        - predict_vs_actual_plot: Scatter plot comparing predictions to actual values
        - vip_plot: Bar chart of Variable Importance in Projection scores
        - vip_table: Table containing VIP scores for each feature

    Configuration:
        - training_design: Defines target variable(s) for the model
        - test_size: Proportion of data for test set (0.0-1.0)
        - number_of_components: Number of PLS components (auto-selected if not specified)
        - scale_data: Whether to standardize features before modeling
    """

    input_specs = InputSpecs({'data': InputSpec(Table, human_name="Data",
                             short_description="Data",)})
    config_specs = ConfigSpecs({
        'training_design': ParamSet(ConfigSpecs({
            'target_name':
            StrParam(
                human_name="Target name",
                short_description="The name of the 'columns' to use as targets.")}),
            human_name="Training design",
            short_description="Define the training design, i.e. the target Y to use for the model."),
            'test_size': FloatParam(
                default_value=0.2,
                min_value=0.0,
                max_value=1.0,
                human_name="Test size",
                short_description="Proportion of the dataset to include in the test split (between 0.0 and 1.0)."),
            'number_of_components': IntParam(
                min_value=1,
                human_name="Number of Components",
                short_description="Number of PLS components to use in the model, if empty, the optimal number will be selected based on MSE.",
                optional=True),
            'scale_data': BoolParam(
                default_value=True,
                human_name="Scale Data",
                short_description="Whether to scale the data before fitting the PLS model.")})
    output_specs = OutputSpecs({'pls_component_plot': OutputSpec(
        PlotlyResource, human_name="PLS Component Plot", short_description="The PLS component plot"),
        'predict_vs_actual_plot': OutputSpec(
            PlotlyResource, human_name="Predicted vs Actual Plot", short_description="The predicted vs actual values plot"),
        'vip_plot': OutputSpec(
            PlotlyResource, human_name="VIP plot", short_description="VIP plot"),
        'vip_table': OutputSpec(
            Table, human_name="VIP Table", short_description="Table of VIP scores"),})


    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        # retrieve the targets name
        dicts_target_name = params['training_design']
        target_columns = []
        for dict_target_name in dicts_target_name:
            target_columns.append(dict_target_name['target_name'])

        # Load data
        df = inputs["data"].get_data()

        # X = all composition + medium parameters
        X = df.drop(columns=target_columns)

        #If X contains a categorical variable, raise an error
        if X.select_dtypes(include=['object', 'category', 'string']).shape[1] > 0:
            raise ValueError("Categorical variables are not supported in PLS Regression. Please encode or remove them.")

        # Y = target variable(s)
        y = df[target_columns]

        # Store column names before scaling
        X_columns = X.columns

        # Split BEFORE scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=20
        )

        # Scaling only if requested
        if params['scale_data']:
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values

        # Determine max components based on training set
        max_components = min(X_train.shape[0], X_train.shape[1])
        mse_list = []

        for n in range(1, max_components+1):
            pls = PLSRegression(n_components=n)
            pls.fit(X_train, y_train)
            Y_pred = pls.predict(X_train)
            mse = mean_squared_error(y_train, Y_pred)
            mse_list.append(mse)

        # Create Plotly plot for MSE vs number of components
        pls_component_plot = go.Figure()
        pls_component_plot.add_trace(go.Scatter(
            x=list(range(1, max_components+1)),
            y=mse_list,
            mode='lines+markers',
            marker=dict(size=8)
        ))
        pls_component_plot.update_layout(
            title="Selecting Optimal Number of PLS Components",
            xaxis_title="Number of PLS Components",
            yaxis_title="MSE"
        )

        # Optimal number of components
        if params['number_of_components'] is None:
            best_n = np.argmin(mse_list) + 1
        else:
            best_n = params['number_of_components']

        #calculate RMSE
        pls = PLSRegression(n_components=best_n)
        pls.fit(X_train, y_train)

        Y_pred = pls.predict(X_test)

        # Evaluate the model performance
        r_squared = pls.score(X_test, y_test)
        mse = mean_squared_error(y_test, Y_pred)

        # Create Plotly plot for predicted vs actual values
        predict_vs_actual_plot = go.Figure()
        predict_vs_actual_plot.add_trace(go.Scatter(
            x=y_test,
            y=Y_pred,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Actual vs Predicted'
        ))
        predict_vs_actual_plot.add_trace(go.Scatter(
            x=[min(y_test), max(y_test)],
            y=[min(y_test), max(y_test)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        predict_vs_actual_plot.update_layout(
            title="PLS Regression: Predicted vs Actual mu",
            xaxis_title="Actual mu",
            yaxis_title="Predicted mu",
            annotations=[dict(
                text=f'R²: {r_squared:.2f}<br>MSE: {mse:.2f}<br>Number of Components: {best_n}',
                xref="paper", yref="paper",
                x=0.05, y=0.95,
                showarrow=False,
                bgcolor="wheat",
                bordercolor="black",
                borderwidth=1
            )]
        )

        # VIP calculation
        vip_scores = self.vip(pls)
        vip_df = pd.DataFrame({'Variable': X_columns, 'VIP': vip_scores})
        # Sort by VIP score
        vip_df = vip_df.sort_values(by='VIP', ascending=False)

        # Create VIP plot with Plotly
        vip_plot = go.Figure()
        vip_plot.add_trace(go.Bar(
            x=vip_df['VIP'],
            y=vip_df['Variable'],
            orientation='h',
            marker=dict(color='skyblue')
        ))
        vip_plot.update_layout(
            title='Variable Importance in Projection (VIP) Scores',
            xaxis_title='VIP Score',
            yaxis_title='Variable',
            yaxis={'categoryorder': 'total ascending'},  # Highest VIP at the top
            height=max(400, len(vip_df) * 25)  # Dynamic height based on number of variables
        )

        return {
            'pls_component_plot': PlotlyResource(pls_component_plot),
            'predict_vs_actual_plot': PlotlyResource(predict_vs_actual_plot),
            'vip_table': Table(vip_df),
            'vip_plot': PlotlyResource(vip_plot)
        }


    def vip(self, pls):
        t = pls.x_scores_
        w = pls.x_weights_
        q = pls.y_loadings_
        p, h = w.shape
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        Wnorm = (w / np.linalg.norm(w, axis=0))
        VIP = np.sqrt(p * (Wnorm**2 @ s) / s.sum())
        return VIP.ravel()
