"""
Script to run complete causal inference analysis in isolated virtual environment.
This script takes a dataframe and target list, performs the entire causal effect analysis,
and outputs results to a folder structure.
"""

import itertools
import os
import pickle
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # Use non-interactive backend

# Constants
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


def run_causal_analysis(input_file: str, output_folder: str):
    """
    Run complete causal inference analysis.

    Args:
        input_file: Path to pickle file containing input data (dataframe + targets)
        output_folder: Path to folder where results will be saved
    """
    # Load input data
    with open(input_file, "rb") as f:
        input_data = pickle.load(f)

    df_numeric = input_data["dataframe"]
    target_all = input_data["targets"]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate all non-empty combinations of targets
    combinations = []
    for r in range(1, len(target_all) + 1):
        combinations.extend(itertools.combinations(target_all, r))

    total_combinations = len(combinations)
    # Process each combination
    for idx, target_subset in enumerate(combinations):
        target_select = list(target_subset)

        # Update progress for combination
        combination_progress = 5 + ((idx / total_combinations) * 90)
        print(
            f"[PROGRESS:{combination_progress:.1f}] Processing combination {idx + 1}/{total_combinations}: {', '.join(target_select)}"
        )

        # Create output subfolder for this combination
        nom_combinaison = COMBINATION_SEPARATOR.join([re.sub(r"\W+", "", t) for t in target_select])
        dossier_output = os.path.join(output_folder, nom_combinaison)
        os.makedirs(dossier_output, exist_ok=True)

        fichier_sortie = os.path.join(dossier_output, "causal_effects.csv")
        fichier_plot = os.path.join(dossier_output, "heatmap_causal_effects.png")

        # Prepare data for this combination
        df_temp = df_numeric.copy()

        # Remove non-selected targets
        for target in target_all:
            if target not in target_select and target in df_temp.columns:
                df_temp = df_temp.drop(target, axis=1)

        # Define variables
        target_vars = [col for col in target_select if col in df_temp.columns]
        treatment_vars = [col for col in df_temp.columns if col not in target_vars]

        results = []

        # Calculate total treatment-target pairs for this combination
        total_pairs = len(treatment_vars) * len(target_vars)
        current_pair = 0

        # Process each treatment-target pair
        for t_name in treatment_vars:
            if t_name not in df_temp.columns:
                continue

            t = df_temp[t_name].values
            if np.all(np.isnan(t)):
                continue

            # Check if treatment is discrete (binary 0/1)
            is_discrete = np.array_equal(np.unique(t[~np.isnan(t)]), [0, 1])
            t_scaled = (
                t if is_discrete else StandardScaler().fit_transform(t.reshape(-1, 1)).ravel()
            )

            # Prepare confounders (other treatments + remaining variables)
            other_treatments = [
                col for col in treatment_vars if col != t_name and col in df_temp.columns
            ]
            x_raw_full = df_temp.drop(columns=[t_name] + target_vars, errors="ignore")
            x_raw_full = df_temp[
                other_treatments + list(x_raw_full.columns.difference(other_treatments))
            ]

            if x_raw_full.shape[1] == 0:
                x_scaled = None
            else:
                # Feature selection using mutual information
                x_raw_full = x_raw_full.fillna(0)
                mi_scores = mutual_info_regression(x_raw_full, t)
                top_idx = np.argsort(mi_scores)[-5:]
                x_selected = x_raw_full.iloc[:, top_idx]
                x_scaled = StandardScaler().fit_transform(x_selected)

            # Process each target
            for y_name in target_vars:
                current_pair += 1
                pair_progress_within_combination = (current_pair / total_pairs) * (
                    90 / total_combinations
                )
                total_progress = combination_progress + pair_progress_within_combination

                print(
                    f"[PROGRESS:{total_progress:.1f}] Analyzing {t_name} → {y_name} (pair {current_pair}/{total_pairs})"
                )

                if y_name not in df_temp.columns:
                    continue

                y = df_temp[y_name].values
                if np.all(np.isnan(y)):
                    continue

                # Scale target
                scaler_y = StandardScaler()
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

                try:
                    # Build and fit causal model
                    model_t = (
                        RandomForestClassifier(n_estimators=100)
                        if is_discrete
                        else RandomForestRegressor(n_estimators=100)
                    )

                    model = (LinearDML if is_discrete else CausalForestDML)(
                        model_y=RandomForestRegressor(n_estimators=100),
                        model_t=model_t,
                        discrete_treatment=is_discrete,
                        random_state=42,
                    )

                    model.fit(y_scaled, t_scaled, X=x_scaled)
                    cate_scaled = model.effect(x_scaled if x_scaled is not None else None)
                    avg_cate_scaled = np.mean(cate_scaled)
                    avg_cate = (
                        avg_cate_scaled * scaler_y.scale_[0]
                        if scaler_y.scale_ is not None
                        else avg_cate_scaled
                    )

                    results.append(
                        {
                            TREATMENT_NAME: t_name,
                            TARGET_NAME: y_name,
                            AVERAGE_CAUSAL_EFFECT_NAME: avg_cate,
                            TYPE_OF_TREATMENT_NAME: MODEL_DISCRETE
                            if is_discrete
                            else MODEL_CONTINUOUS,
                            MODEL_NAME: MODEL_LINEAR_DML
                            if is_discrete
                            else MODEL_CAUSAL_FOREST_DML,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            TREATMENT_NAME: t_name,
                            TARGET_NAME: y_name,
                            AVERAGE_CAUSAL_EFFECT_NAME: np.nan,
                            ERROR_NAME: str(e),
                        }
                    )

        # Export results
        df_results = pd.DataFrame(results)
        df_results.to_csv(fichier_sortie, index=False)

        # Generate heatmap
        heatmap_data = df_results.pivot(
            index=TARGET_NAME, columns=TREATMENT_NAME, values=AVERAGE_CAUSAL_EFFECT_NAME
        )
        if not heatmap_data.empty:
            fig_width = max(12, 0.5 * len(heatmap_data.columns))
            plt.figure(figsize=(fig_width, 8))
            sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            plt.title(f"Heatmap - {TARGET_NAME}s: {', '.join(target_select)}")
            plt.xlabel(TREATMENT_NAME)
            plt.ylabel(TARGET_NAME)
            plt.tight_layout()
            plt.savefig(fichier_plot, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python _run_causal_analysis.py <input_file> <output_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    run_causal_analysis(input_file, output_folder)
