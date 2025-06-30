import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from gws_core import (Settings, ConfigParams, InputSpec,
                      InputSpecs, ListParam, OutputSpec, OutputSpecs, ConfigSpecs,
                      Table, Task, TaskInputs,
                      TaskOutputs, TypingStyle, task_decorator, Folder)


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
        'targets': ListParam(human_name= "Target(s)", short_description="Target(s)",)})
    output_specs = OutputSpecs({
        'results_folder': OutputSpec(Folder, human_name="Results folder", short_description="The folder containing the results"),

    })


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

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:

        # Load and prepare data
        df = inputs["data"].get_data()
        df_numeric = df.select_dtypes(include='number').dropna()

        folder_output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), Settings.make_temp_dir())

        # Define targets
        target_all = params.get('targets')

        # Générer toutes les combinaisons non vides de target_all
        combinations = []
        for r in range(1, len(target_all) + 1):
            combinations.extend(itertools.combinations(target_all, r))

        # Boucle sur chaque combinaison de cibles
        for idx, target_subset in enumerate(combinations):
            target_select = list(target_subset)

            # Créer le dossier de sortie
            nom_combinaison = "_".join([re.sub(r'\W+', '', t) for t in target_select])
            dossier_output = os.path.join(folder_output_path, nom_combinaison)
            os.makedirs(dossier_output, exist_ok=True)

            fichier_sortie = os.path.join(dossier_output, "causal_effects.csv")
            fichier_plot = os.path.join(dossier_output, "heatmap_causal_effects.png")

            # Préparer les données pour cette combinaison
            df_temp = df_numeric.copy()

            # Supprimer les targets non sélectionnées
            for target in target_all:
                if target not in target_select and target in df_temp.columns:
                    df_temp = df_temp.drop(target, axis=1)

            # Définir les variables
            target_vars = [col for col in target_select if col in df_temp.columns]
            treatment_vars = [col for col in df_temp.columns if col not in target_vars]

            results = []

            for T_name in treatment_vars:
                if T_name not in df_temp.columns:
                    continue

                T = df_temp[T_name].values
                if np.all(np.isnan(T)):
                    continue

                is_discrete = np.array_equal(np.unique(T[~np.isnan(T)]), [0, 1])
                T_scaled = T if is_discrete else StandardScaler().fit_transform(T.reshape(-1, 1)).ravel()

                other_treatments = [col for col in treatment_vars if col != T_name and col in df_temp.columns]
                X_raw_full = df_temp.drop(columns=[T_name] + target_vars, errors='ignore')
                X_raw_full = df_temp[other_treatments + list(X_raw_full.columns.difference(other_treatments))]

                if X_raw_full.shape[1] == 0:
                    X_scaled = None
                else:
                    X_raw_full = X_raw_full.fillna(0)
                    mi_scores = mutual_info_regression(X_raw_full, T)
                    top_idx = np.argsort(mi_scores)[-5:]
                    X_selected = X_raw_full.iloc[:, top_idx]
                    X_scaled = StandardScaler().fit_transform(X_selected)

                for Y_name in target_vars:
                    if Y_name not in df_temp.columns:
                        continue

                    Y = df_temp[Y_name].values
                    if np.all(np.isnan(Y)):
                        continue

                    scaler_Y = StandardScaler()
                    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()

                    try:
                        model_t = RandomForestClassifier(n_estimators=100) if is_discrete else RandomForestRegressor(n_estimators=100)
                        model = (LinearDML if is_discrete else CausalForestDML)(
                            model_y=RandomForestRegressor(n_estimators=100),
                            model_t=model_t,
                            discrete_treatment=is_discrete,
                            random_state=42
                        )

                        model.fit(Y_scaled, T_scaled, X=X_scaled)
                        cate_scaled = model.effect(X_scaled if X_scaled is not None else None)
                        avg_cate_scaled = np.mean(cate_scaled)
                        avg_cate = avg_cate_scaled * scaler_Y.scale_[0]

                        results.append({
                            self.TREATMENT_NAME: T_name,
                            self.TARGET_NAME: Y_name,
                            self.AVERAGE_CAUSAL_EFFECT_NAME: avg_cate,
                            self.TYPE_OF_TREATMENT_NAME: self.MODEL_DISCRETE if is_discrete else self.MODEL_CONTINUOUS,
                            self.MODEL_NAME: self.MODEL_LINEAR_DML if is_discrete else self.MODEL_CAUSAL_FOREST_DML,
                        })
                    except Exception as e:
                        results.append({
                            self.TREATMENT_NAME: T_name,
                            self.TARGET_NAME: Y_name,
                            self.AVERAGE_CAUSAL_EFFECT_NAME: np.nan,
                            self.ERROR_NAME: str(e)
                        })

            # Export des résultats
            df_results = pd.DataFrame(results)
            df_results.to_csv(fichier_sortie, index=False)

            # Génération de la heatmap
            heatmap_data = df_results.pivot(index=self.TARGET_NAME, columns=self.TREATMENT_NAME, values=self.AVERAGE_CAUSAL_EFFECT_NAME)
            if not heatmap_data.empty:
                fig_width = max(12, 0.5 * len(heatmap_data.columns))
                plt.figure(figsize=(fig_width, 8))
                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, fmt=".2f")
                plt.title(f"Heatmap - {self.TARGET_NAME}s: {', '.join(target_select)}")
                plt.xlabel(self.TREATMENT_NAME)
                plt.ylabel(self.TARGET_NAME)
                plt.tight_layout()
                plt.savefig(fichier_plot, bbox_inches="tight")
                plt.close()

        folder_results = Folder(folder_output_path)
        folder_results.name = "Causal Effects Results"
        return {
            'results_folder': folder_results,
        }

