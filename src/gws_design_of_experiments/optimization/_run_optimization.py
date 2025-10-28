"""
Script to run complete optimization analysis in isolated virtual environment.
This script takes preprocessed data and parameters, performs model training and optimization,
and outputs results to a folder structure.
"""

import multiprocessing
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class MilieuOptimizationProblem(ElementwiseProblem):
    """Problem definition for pymoo optimization."""

    def __init__(
            self, model, low, up, targets, all_outputs, output_thresholds, X_columns, add_cv=True,
            elementwise_runner=None):
        self.targets = targets
        self.add_cv = add_cv
        self.all_outputs = all_outputs
        self.thresholds = output_thresholds
        self.X_columns = X_columns
        n_obj = len(targets) + (1 if add_cv else 0)

        super().__init__(
            n_var=len(low), n_obj=n_obj, n_constr=0,
            xl=low, xu=up,
            elementwise_evaluation=True,
            elementwise_runner=elementwise_runner
        )
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        x_df = pd.DataFrame([x], columns=self.X_columns)
        y = self.model.predict(x_df)[0]
        output_dict = dict(zip(self.all_outputs, y))
        cv = sum(max(0, self.thresholds[t] - output_dict[t]) for t in self.targets) if self.add_cv else 0
        objectives = [-output_dict[t] for t in self.targets] + ([cv] if self.add_cv else [])
        out["F"] = objectives


class MyCallback(Callback):
    """Callback to track optimization progress."""

    def __init__(self, output_dir: str, save_every=50, iterations=None):
        super().__init__()
        self.save_every = save_every
        self.history = []
        self.output_dir = output_dir
        self.iterations = iterations

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        best_cv = F[:, -1].min()
        gen = algorithm.n_gen
        self.history.append({"generation": gen, "best_cv": best_cv})

        # Update progress during optimization (65% to 95% range)
        if self.iterations:
            if gen % 10 == 0 or gen == self.iterations:
                progress = 65 + (gen / self.iterations) * 30
                print(f"[PROGRESS:{progress:.1f}] Optimization generation {gen}/{self.iterations}")

        if gen % self.save_every == 0:
            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "optimization_progress.csv"), index=False)


def optimize_model(name, base_model, param_dist, scorer, X, Y, kf, pipelines, n_iter=200):
    """Optimize a single model using RandomizedSearchCV."""
    # Calculate total search space size
    import operator
    from functools import reduce
    search_space_size = reduce(operator.mul, [len(v) for v in param_dist.values()], 1)

    # Adjust n_iter to not exceed search space size
    actual_n_iter = min(n_iter, search_space_size)

    search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=actual_n_iter,
                                scoring='r2', cv=3, random_state=42, n_jobs=-1)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(search))
    ])
    score = cross_val_score(pipe, X, Y, cv=kf, scoring=scorer)
    pipelines.append((np.mean(score), pipe, name))
    return pipelines


def run_optimization(input_file: str, output_folder: str):
    """
    Run complete optimization analysis.

    Args:
        input_file: Path to pickle file containing input data and parameters
        output_folder: Path to folder where results will be saved
    """
    print("[PROGRESS:0] Starting optimization...")
    print("[INFO] Loading input data...")

    # Load input data
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)

    # Extract all parameters
    data_filtered = input_data['data_filtered']
    optimization_targets = input_data['optimization_targets']
    full_thresholds = input_data['full_thresholds']
    output_thresholds = input_data['output_thresholds']
    population_size = input_data['population_size']
    iterations = input_data['iterations']
    manual_constraints = input_data['manual_constraints']
    add_cv_as_penalty = input_data.get('add_cv_as_penalty', True)

    print("[PROGRESS:5] Preparing data...")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Prepare data
    Y = data_filtered[optimization_targets]
    X = data_filtered.drop(columns=optimization_targets)

    # Remove zero columns
    zero_cols = X.columns[(X == 0).all()].tolist()
    if zero_cols:
        X = X.drop(columns=zero_cols)
        print(f"[INFO] Removed {len(zero_cols)} zero-variance columns")

    print(f"[INFO] Training data shape: X={X.shape}, Y={Y.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(lambda y_true, y_pred: r2_score(y_true, y_pred, multioutput='uniform_average'))
    pipelines = []

    print("[PROGRESS:10] Starting model optimization...")
    print("[PROGRESS:15] Optimizing RandomForest model...")
    pipelines = optimize_model("RandomForest", RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200, 400],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt']
    }, scorer, X, Y, kf, pipelines)

    print("[PROGRESS:25] Optimizing XGBoost model...")
    pipelines = optimize_model("XGBoost", XGBRegressor(random_state=42), {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.2]
    }, scorer, X, Y, kf, pipelines)

    print("[PROGRESS:40] Optimizing CatBoost model...")
    pipelines = optimize_model("CatBoost", CatBoostRegressor(random_seed=42, verbose=0), {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.2]
    }, scorer, X, Y, kf, pipelines)

    print("[PROGRESS:55] Selecting best model and preparing optimization...")
    best_score, pipeline, model_name = max(pipelines, key=lambda x: x[0])
    print(f"[INFO] Best model: {model_name} (R² score: {best_score:.4f})")

    print("[INFO] Training final model on full dataset...")
    pipeline.fit(X, Y)

    # Save predictions
    predictions = pipeline.predict(X)
    df_pred = pd.DataFrame(predictions, columns=Y.columns)
    df_actual_vs_pred = pd.concat([Y.reset_index(drop=True), df_pred.add_suffix('_pred')], axis=1)
    df_actual_vs_pred.to_csv(os.path.join(output_folder, "actual_vs_predicted.csv"), index=False)
    print("[INFO] Saved actual vs predicted values")

    # Save feature importance
    multi_output_model = pipeline.named_steps['model']
    importance_matrix = [est.best_estimator_.feature_importances_ for est in multi_output_model.estimators_]
    importance_matrix = np.array(importance_matrix)

    df_importance = pd.DataFrame(importance_matrix.T, index=X.columns, columns=Y.columns)
    df_importance.to_csv(os.path.join(output_folder, "feature_importance_matrix.csv"))
    print("[INFO] Saved feature importance matrix")

    # Prepare constraints
    low_mod = np.min(X.values, axis=0).copy()
    up_mod = np.max(X.values, axis=0).copy()

    for feature, bounds in manual_constraints.items():
        if feature in X.columns:
            idx = X.columns.get_loc(feature)
            low_mod[idx] = bounds.get('lower_bound')
            up_mod[idx] = bounds.get('upper_bound')

    constraints_df = pd.DataFrame({"Feature": X.columns, "Lower_Bound_Used": low_mod, "Upper_Bound_Used": up_mod})
    constraints_df.to_csv(os.path.join(output_folder, "constraints_used_in_optimization.csv"), index=False)
    print("[INFO] Saved constraints configuration")

    print("[PROGRESS:60] Setting up optimization problem...")

    # Setup parallel processing
    n_proccess = 16
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)
    print(f"[INFO] Using {n_proccess} parallel processes")

    # Create optimization problem
    problem = MilieuOptimizationProblem(
        pipeline, low_mod, up_mod,
        targets=optimization_targets,
        all_outputs=Y.columns.tolist(),
        output_thresholds=output_thresholds,
        X_columns=X.columns,
        add_cv=add_cv_as_penalty,
        elementwise_runner=runner
    )

    # Select algorithm
    if len(optimization_targets) == 1 and not add_cv_as_penalty:
        algorithm = GA(pop_size=population_size)
        termination = get_termination("n_gen", iterations)
        print(f"[INFO] Using GA algorithm with population={population_size}, iterations={iterations}")
    else:
        algorithm = NSGA2(
            pop_size=population_size,
            sampling=LHS(),
            crossover=SBX(prob=0.95, eta=3),
            mutation=PM(eta=3),
            eliminate_duplicates=True
        )
        termination = get_termination("n_gen", iterations)
        print(f"[INFO] Using NSGA2 algorithm with population={population_size}, iterations={iterations}")

    print("[PROGRESS:65] Starting optimization algorithm...")
    res = minimize(problem, algorithm, termination, seed=42, verbose=True,
                   callback=MyCallback(output_dir=output_folder, save_every=10, iterations=iterations))
    pool.close()
    pool.join()

    print("[PROGRESS:95] Saving optimization results...")

    if res.F is not None and len(res.F) > 0:
        print(f"[INFO] Found {len(res.F)} optimization solutions")

        df_resultats = pd.DataFrame(res.X, columns=X.columns)
        df_resultats['CV'] = res.F[:, -1]
        for i, t in enumerate(optimization_targets):
            df_resultats[t] = -res.F[:, i]

        df_resultats['CV_percent'] = (df_resultats['CV'] / sum(full_thresholds.values()))*100
        # Drop CV column
        df_resultats = df_resultats.drop(columns=['CV'])

        for target in optimization_targets:
            if target in full_thresholds:
                df_resultats[target + '_percent'] = (df_resultats[target] / full_thresholds[target])*100

        df_resultats.to_csv(os.path.join(output_folder, "generalized_solutions.csv"), index=False)
        print("[INFO] Saved all optimization solutions")

        df_best = df_resultats.sort_values(by=['CV_percent'] + optimization_targets,
                                           ascending=[False] + [False]*len(optimization_targets)).iloc[[0]]
        df_best.to_csv(os.path.join(output_folder, "best_generalized_solution.csv"), index=False)
        print("[INFO] Saved best solution")

        # Print best solution summary
        print(f"[INFO] Best solution CV: {df_best['CV_percent'].values[0]:.2f}%")
        for target in optimization_targets:
            if target + '_percent' in df_best.columns:
                print(f"[INFO] Best {target}: {df_best[target + '_percent'].values[0]:.2f}%")
    else:
        print("[WARNING] No optimization solutions found")

    print("[PROGRESS:100] Optimization completed!")
    print("[SUCCESS] Optimization completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python _run_optimization.py <input_file> <output_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    run_optimization(input_file, output_folder)
