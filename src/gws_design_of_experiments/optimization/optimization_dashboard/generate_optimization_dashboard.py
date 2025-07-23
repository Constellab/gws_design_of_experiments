from gws_core import (ConfigParams, AppConfig, AppType, OutputSpec,
                      OutputSpecs, StreamlitResource, Task, TaskInputs,
                      TaskOutputs, app_decorator, task_decorator,
                      InputSpecs, ConfigSpecs, InputSpec, Folder)


@app_decorator("OptimizationDashboardAppConfig", app_type=AppType.STREAMLIT,
               human_name="Generate show case app")
class OptimizationDashboardAppConfig(AppConfig):

    # retrieve the path of the app folder, relative to this file
    # the app code folder starts with a underscore to avoid being loaded when the brick is loaded
    def get_app_folder_path(self):
        return self.get_app_folder_from_relative_path(__file__, "_optimization_dashboard")


@task_decorator("GenerateOptimizationDashboard", human_name="Generate OptimizationDashboard app",
                style=StreamlitResource.copy_style())
class GenerateOptimizationDashboard(Task):
    """
    Task that generates an interactive Streamlit dashboard for optimization results visualization.

    This task creates a comprehensive dashboard that allows users to explore and analyze
    optimization results through multiple interactive views including:

    - **Summary Best Solution**: Displays the best optimization solution found
    - **3D Surface Explorer**: Interactive 3D visualization with polynomial fitting
    - **Feature Importance Matrix**: Visualizes the importance of different features
    - **Observed vs Predicted**: Scatter plots comparing actual vs predicted values
    - **Data Explorer**: Interactive data table with sorting and filtering capabilities

    The dashboard expects optimization results to be stored in a folder containing:
    - `generalized_solutions.csv`: All optimization solutions
    - `best_generalized_solution.csv`: The best solution found
    - `actual_vs_predicted.csv`: Observed vs predicted values (optional)
    - `feature_importance_matrix.csv`: Feature importance data (optional)

    Inputs:
        folder (Folder): Directory containing optimization result CSV files

    Outputs:
        streamlit_app (StreamlitResource): Interactive Streamlit dashboard application

    Usage:
        This task is typically used after running optimization algorithms to provide
        an interactive way to explore and understand the results. The generated
        dashboard helps users identify patterns, validate models, and make informed
        decisions based on the optimization outcomes.
    """

    input_specs = InputSpecs({'folder': InputSpec(Folder)})
    output_specs = OutputSpecs({
        'streamlit_app': OutputSpec(StreamlitResource)
    })

    config_specs = ConfigSpecs({})

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        """ Run the task """

        streamlit_app = StreamlitResource()

        # set the input in the streamlit resource
        folder: Folder = inputs.get('folder')
        streamlit_app.add_resource(folder, create_new_resource=False)

        streamlit_app.set_app_config(OptimizationDashboardAppConfig())
        streamlit_app.name = "Optimization Dashboard"

        return {"streamlit_app": streamlit_app}
