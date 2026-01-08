import os
import pickle

from gws_core import (
    ConfigParams,
    ConfigSpecs,
    Folder,
    InputSpec,
    InputSpecs,
    IntParam,
    JSONDict,
    ListParam,
    MambaShellProxy,
    OutputSpec,
    OutputSpecs,
    ParamSet,
    StrParam,
    Table,
    Task,
    TaskInputs,
    TaskOutputs,
    TypingStyle,
    task_decorator,
)

from .optimization_env_helper import OptimizationEnvHelper


@task_decorator(
    "Optimization",
    human_name="Optimization",
    short_description="Optimization with virtual environment",
    style=TypingStyle.material_icon(material_icon_name="auto_mode", background_color="#5acce9"),
)
class Optimization(Task):
    """
    Optimization task using machine learning models in isolated virtual environment.

    This task performs optimization on experimental data by:
    1. Training multiple machine learning models (Random Forest, XGBoost, CatBoost)
    2. Selecting the best performing model based on cross-validation R² scores
    3. Using algorithms (NSGA-II or GA) to find optimal solutions
    4. Generating comprehensive optimization results and analysis files

    The optimization process considers:
    - **Target variables**: Variables to maximize during optimization
    - **Constraints**: Manual bounds on input features
    - **Thresholds**: Minimum acceptable values for target variables

    **Generated Output Files:**
    - `generalized_solutions.csv`: All optimization solutions found
    - `best_generalized_solution.csv`: Best solution based on CV and target values
    - `actual_vs_predicted.csv`: Model validation data (observed vs predicted)
    - `feature_importance_matrix.csv`: Feature importance for each target variable
    - `constraints_used_in_optimization.csv`: Bounds applied to each feature
    - `optimization_progress.csv`: Convergence history during optimization

    **Inputs:**
        data (Table): Experimental data containing features and target variables
        targets_thresholds (JSONDict): Minimum threshold values for each target variable
        manual_constraints (JSONDict): Custom bounds for input features in format:
                                     {"feature_name": {"lower_bound": value, "upper_bound": value}}

    **Outputs:**
        results_folder (Folder): Directory containing all optimization results and analysis files

    **Example:**
        For a chemical process optimization, you might want to maximize yield and purity
        while keeping temperature below 100°C and pressure above 2 bar, with minimum
        yield of 80% and minimum purity of 95%.
    """

    input_specs = InputSpecs(
        {
            "data": InputSpec(
                Table,
                human_name="Data",
                short_description="Data",
            ),
            "manual_constraints": InputSpec(
                JSONDict,
                human_name="Manual constraints",
                short_description="Manual constraints for optimization",
            ),
        }
    )
    config_specs = ConfigSpecs(
        {
            "population_size": IntParam(
                default_value=500,
                optional=False,
                human_name="Population size",
                short_description="Population size for the optimization algorithm",
            ),
            "iterations": IntParam(default_value=100, optional=False, human_name="Iterations"),
            "columns_to_exclude": ListParam(
                human_name="Columns to Exclude",
                short_description="List of column names to exclude from optimization analysis",
                optional=True,
            ),
            "targets_thresholds": ParamSet(
                ConfigSpecs(
                    {
                        "targets": StrParam(
                            default_value=None,
                            optional=False,
                            human_name="Target",
                            short_description="Target to optimize",
                        ),
                        "thresholds": IntParam(
                            default_value=None,
                            optional=False,
                            human_name="Objective",
                            short_description="Objective value for the target",
                        ),
                    }
                ),
                optional=False,
                human_name="Targets with objectives",
                short_description="Targets to optimize and their objective values",
            ),
        }
    )

    output_specs = OutputSpecs(
        {
            "results_folder": OutputSpec(
                Folder,
                human_name="Results folder",
                short_description="The folder containing the results",
            ),
        }
    )

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        """Run the optimization task in virtual environment."""
        # Create shell proxy for optimization environment
        shell_proxy: MambaShellProxy = OptimizationEnvHelper.create_proxy(self.message_dispatcher)

        # Get inputs
        data_filtered = inputs["data"].get_data()
        manual_constraints = inputs["manual_constraints"].get_data()

        # Get columns to exclude from config
        columns_to_exclude = params.get("columns_to_exclude")
        if columns_to_exclude:
            data_filtered = data_filtered.drop(columns=columns_to_exclude, errors="ignore")

        # Get parameters
        population_size = params.get("population_size")
        iterations = params.get("iterations")
        full_thresholds = {
            target["targets"]: target["thresholds"] for target in params.get("targets_thresholds")
        }

        # Prepare data
        columns_data = data_filtered.columns.tolist()
        manual_constraints_name = list(manual_constraints.keys())

        # Define targets
        optimization_targets = list(full_thresholds.keys())
        optimization_targets_all = [
            col for col in columns_data if col not in manual_constraints_name
        ]

        # Filter thresholds for selected targets
        output_thresholds = {k: full_thresholds[k] for k in optimization_targets}

        # Remove non-selected targets
        for target in optimization_targets_all:
            if target not in optimization_targets and target in data_filtered.columns:
                data_filtered = data_filtered.drop(target, axis=1)

        data_filtered = data_filtered.dropna()

        # Prepare input data for virtual environment
        input_data = {
            "data_filtered": data_filtered,
            "optimization_targets": optimization_targets,
            "full_thresholds": full_thresholds,
            "output_thresholds": output_thresholds,
            "population_size": population_size,
            "iterations": iterations,
            "manual_constraints": manual_constraints,
            "add_cv_as_penalty": True,
        }

        # Create temporary files for input/output
        input_file = os.path.join(shell_proxy.working_dir, "optimization_input.pkl")
        output_folder = os.path.join(shell_proxy.working_dir, "optimization_results")

        # Save input data
        self.log_info_message("Saving input data for virtual environment")
        with open(input_file, "wb") as f:
            pickle.dump(input_data, f)

        # Get path to the optimization script
        script_path = os.path.join(os.path.dirname(__file__), "_run_optimization.py")

        # Run the complete optimization in virtual environment
        cmd = ["python", script_path, input_file, output_folder]
        result_code = shell_proxy.run(cmd, dispatch_stdout=True, dispatch_stderr=True)

        if result_code != 0:
            self.log_error_message("Failed to run optimization in virtual environment")
            raise Exception("Optimization failed in virtual environment")

        # The output_folder now contains all the CSV files
        # Return it directly as the Folder resource
        folder_results = Folder(output_folder)
        folder_results.name = "Optimization Results"

        return {
            "results_folder": folder_results,
        }
