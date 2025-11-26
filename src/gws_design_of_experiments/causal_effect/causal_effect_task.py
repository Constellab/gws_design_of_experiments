import os
import pickle

from gws_core import (ConfigParams, ConfigSpecs, Folder, InputSpec, InputSpecs,
                      ListParam, MambaShellProxy, OutputSpec, OutputSpecs,
                      Table, Task, TaskInputs, TaskOutputs, TypingStyle,
                      task_decorator)

from .econml_env_helper import EconmlEnvHelper


@task_decorator("CausalEffect", human_name="Causal Effect", short_description="Causal Effect",
                style=TypingStyle.material_icon(material_icon_name="gradient",
                                                background_color="#f17093"))
class CausalEffect(Task):
    """
    CausalEffect task performs causal inference analysis to estimate the causal effects
    of treatments on target variables using machine learning methods.

    This task implements causal effect estimation using LinearDML for discrete treatments and CausalForestDML for continuous
    treatments. It analyzes all possible combinations of the specified target variables and
    estimates the Average Treatment Effect (ATE) for each treatment-target pair.

    **What this task does:**
    - Identifies treatment from your dataset
    - For each treatment-target combination, estimates the causal effect using appropriate DML models
    - Handles both discrete and continuous treatment variables
    - Uses feature selection based on mutual information to select relevant confounders
    - Generates results for all possible combinations of target variables
    - Creates heatmap visualizations of causal effects
    - Exports results as CSV files and PNG heatmaps

    **Input Requirements:**
    - **Data**: A Table containing numerical data with:
      - At least one target variable
      - At least one treatment variable
      - Optional: Additional variables that serve as confounders/covariates
      - All variables should be numerical
      - Missing values will be handled automatically

    **Configuration:**
    - **targets**: List of column names from your data that represent target variables
      These are the variables for which you want to measure causal effects

    **Outputs:**
    - **results_folder**: A folder containing:
        - A subfolder for each combination of target variables
            - Each subfolder contains:
            - CSV file with causal effect estimates for each treatment-target pair
            - Heatmap visualization of the causal effects for that combination


    **Example Use Case:**
    If you have data with variables like 'drug_dose', 'exercise_hours', 'blood_pressure', 'cholesterol'
    and you specify targets=['blood_pressure', 'cholesterol'], the task will:
    1. Treat 'drug_dose' and 'exercise_hours' as treatments
    2. Estimate how each treatment affects each target
    3. Generate results for individual targets and their combination
    4. Create visualizations showing the strength and direction of causal effects

    **Note:** The task uses sophisticated causal inference methods that account for confounding
    variables automatically, but results should be interpreted carefully considering domain knowledge
    and potential unmeasured confounders.
    """

    input_specs = InputSpecs({'data': InputSpec(Table, human_name="Data",
                             short_description="Data",)})
    config_specs = ConfigSpecs({
        'targets': ListParam(human_name="Target(s)", short_description="Target(s)",)})
    output_specs = OutputSpecs({'results_folder': OutputSpec(
        Folder, human_name="Results folder", short_description="The folder containing the results"), })

    TREATMENT_NAME = "Treatment"
    TARGET_NAME = "Target"
    AVERAGE_CAUSAL_EFFECT_NAME = "Average Causal Effect (CATE)"
    TYPE_OF_TREATMENT_NAME = "Type of treatment"
    MODEL_NAME = "Model"
    MODEL_DISCRETE = "Discrete"
    MODEL_CONTINUOUS = "Continuous"
    MODEL_LINEAR_DML = "LinearDML"
    MODEL_CAUSAL_FOREST_DML = "CausalForestDML"
    ERROR_NAME = "Error"
    COMBINATION_SEPARATOR = "|"

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        # Create shell proxy for econml environment
        shell_proxy: MambaShellProxy = EconmlEnvHelper.create_proxy(
            self.message_dispatcher)

        self.update_progress_value(5, message="Preparing data for virtual environment")

        # Load and prepare data
        df = inputs["data"].get_data()
        df_numeric = df.select_dtypes(include='number').dropna()

        # Get targets from config
        target_all = params.get('targets')

        # Prepare input data for virtual environment
        input_data = {
            'dataframe': df_numeric,
            'targets': target_all
        }

        # Create temporary files for input/output
        input_file = os.path.join(shell_proxy.working_dir, "causal_input.pkl")
        output_folder = os.path.join(shell_proxy.working_dir, "causal_results")

        # Save input data
        self.log_info_message("Saving input data for virtual environment")
        with open(input_file, 'wb') as f:
            pickle.dump(input_data, f)

        # Get path to the analysis script
        script_path = os.path.join(os.path.dirname(__file__), "_run_causal_analysis.py")

        self.update_progress_value(10, message="Running causal analysis in virtual environment")

        # Run the complete analysis in virtual environment
        cmd = ["python", script_path, input_file, output_folder]
        result_code = shell_proxy.run(cmd, dispatch_stderr=True, dispatch_stdout=True)

        if result_code != 0:
            self.log_error_message("Failed to run causal analysis in virtual environment")
            raise Exception("Causal analysis failed in virtual environment")

        self.update_progress_value(95, message="Analysis completed, retrieving results")

        # The output_folder now contains all the CSV and PNG files organized in subfolders
        # Return it directly as the Folder resource
        folder_results = Folder(output_folder)
        folder_results.name = "Causal Effects Results"

        self.update_progress_value(100, message="Causal effect analysis completed")

        return {
            'results_folder': folder_results,
        }
