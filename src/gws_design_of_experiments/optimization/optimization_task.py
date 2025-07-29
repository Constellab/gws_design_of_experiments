import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.callback import Callback
from pymoo.core.problem import StarmapParallelization

import multiprocessing

from gws_core import (ConfigParams, InputSpec, InputSpecs, OutputSpec,
                      OutputSpecs, Task, TaskInputs, TaskOutputs, ListParam,
                      task_decorator, ConfigSpecs, TypingStyle, Table, Folder, JSONDict, ParamSet, StrParam, IntParam)


@task_decorator("Optimization", human_name="Optimization", short_description="Optimization",
                style=TypingStyle.material_icon(material_icon_name="auto_mode",
                                                background_color="#5acce9"))
class Optimization(Task):
    """
    Optimization task using machine learning models.

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

    input_specs = InputSpecs({'data': InputSpec(Table, human_name="Data", short_description="Data",),
                            'manual_constraints': InputSpec(JSONDict, human_name="Manual constraints",
                            short_description="Manual constraints for optimization")})
    config_specs = ConfigSpecs({
        'population_size': IntParam(default_value=500, optional=False, human_name="Population size", short_description="Population size for the optimization algorithm"),
        'iterations': IntParam(default_value=100, optional=False, human_name="Iterations"),
        'targets_thresholds': ParamSet(ConfigSpecs({
            'targets': StrParam(default_value=None, optional=False, human_name="Target", short_description="Target to optimize"),
            'thresholds': IntParam(default_value=None, optional=False, human_name="Objective", short_description="Objective value for the target"),
        }), optional=False, human_name="Targets with objectives", short_description="Targets to optimize and their objective values")
        })

    output_specs = OutputSpecs({'results_folder': OutputSpec(
        Folder, human_name="Results folder", short_description="The folder containing the results"), })


    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        """ Run the task """
        ## Get inputs
        data_filtered = inputs["data"].get_data()
        # Full threshold dictionary
        full_thresholds = {target['targets']: target['thresholds'] for target in params.get("targets_thresholds")}

        # Get parameters
        population_size = params.get("population_size")
        iterations = params.get("iterations")
        #Manual constraints
        manual_constraints = inputs["manual_constraints"].get_data()

        columns_data = data_filtered.columns.tolist()
        manual_constraints_name = list(manual_constraints.keys())
        # Define targets
        optimization_targets = list(full_thresholds.keys())
        optimization_targets_all = [col for col in columns_data if col not in manual_constraints_name]

        # Filter thresholds for selected targets
        output_thresholds = {k: full_thresholds[k] for k in optimization_targets}
        add_cv_as_penalty = True

        # --- Paths and data loading ---
        output_dir = 'output_dir'
        os.makedirs(output_dir, exist_ok=True)



        # Supprimer les targets non sélectionnés
        for target in optimization_targets_all:
            if target not in optimization_targets and target in data_filtered.columns:
                data_filtered = data_filtered.drop(target, axis=1)

        data_filtered=data_filtered.dropna()


        Y = data_filtered[optimization_targets]
        X = data_filtered.drop(columns=optimization_targets)

        zero_cols = X.columns[(X == 0).all()].tolist()
        if zero_cols:
            X = X.drop(columns=zero_cols)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(lambda y_true, y_pred: r2_score(y_true, y_pred, multioutput='uniform_average'))
        pipelines = []

        self.update_progress_value(5, message="Preparing data and models")

        self.update_progress_value(10, message="Optimizing RandomForest model")
        pipelines = self.optimize_model("RandomForest", RandomForestRegressor(random_state=42), {
            'n_estimators': [100, 200, 400],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt']
        },scorer, X, Y, kf, pipelines)

        self.update_progress_value(25, message="Optimizing XGBoost model")
        pipelines = self.optimize_model("XGBoost", XGBRegressor(random_state=42), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.2]
        },scorer, X, Y, kf, pipelines)

        self.update_progress_value(40, message="Optimizing CatBoost model")
        pipelines = self.optimize_model("CatBoost", CatBoostRegressor(random_seed=42, verbose=0), {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.2]
        },scorer, X, Y, kf, pipelines)

        self.update_progress_value(55, message="Selecting best model and preparing optimization")
        best_score, pipeline, model_name = max(pipelines, key=lambda x: x[0])
        pipeline.fit(X, Y)

        predictions = pipeline.predict(X)
        df_pred = pd.DataFrame(predictions, columns=Y.columns)
        df_actual_vs_pred = pd.concat([Y.reset_index(drop=True), df_pred.add_suffix('_pred')], axis=1)
        df_actual_vs_pred.to_csv(os.path.join(output_dir, "actual_vs_predicted.csv"), index=False)

        multi_output_model = pipeline.named_steps['model']
        importance_matrix = [est.best_estimator_.feature_importances_ for est in multi_output_model.estimators_]
        importance_matrix = np.array(importance_matrix)

        df_importance = pd.DataFrame(importance_matrix.T, index=X.columns, columns=Y.columns)
        df_importance.to_csv(os.path.join(output_dir, "feature_importance_matrix.csv"))

        low_mod = np.min(X.values, axis=0).copy()
        up_mod = np.max(X.values, axis=0).copy()

        for feature, bounds in manual_constraints.items():
            if feature in X.columns:
                idx = X.columns.get_loc(feature)
                low_mod[idx] = bounds.get('lower_bound')
                up_mod[idx] = bounds.get('upper_bound')

        constraints_df = pd.DataFrame({"Feature": X.columns, "Lower_Bound_Used": low_mod, "Upper_Bound_Used": up_mod})
        constraints_df.to_csv(os.path.join(output_dir, "constraints_used_in_optimization.csv"), index=False)


        n_proccess = 16
        pool = multiprocessing.Pool(n_proccess)
        runner = StarmapParallelization(pool.starmap)

        problem = MilieuOptimizationProblem(
            pipeline, low_mod, up_mod,
            targets=optimization_targets,
            all_outputs=Y.columns.tolist(),
            output_thresholds=output_thresholds,
            X_columns=X.columns,
            add_cv=add_cv_as_penalty,
            elementwise_runner=runner
        )

        if len(optimization_targets) == 1 and not add_cv_as_penalty:
            algorithm = GA(pop_size=population_size)
            termination = get_termination("n_gen", iterations)
        else:
            algorithm = NSGA2(
                pop_size=population_size,
                sampling=LHS(),
                crossover=SBX(prob=0.95, eta=3),
                mutation=PM(eta=3),
                eliminate_duplicates=True
            )
            termination = get_termination("n_gen", iterations)

        self.update_progress_value(65, message="Starting optimization algorithm")
        res = minimize(problem, algorithm, termination, seed=42, verbose=True, callback=MyCallback(output_dir=output_dir, save_every=10, task=self, iterations=iterations))
        pool.close()
        pool.join()

        self.update_progress_value(95, message="Saving optimization results")
        if res.F is not None and len(res.F) > 0:
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


            df_resultats.to_csv(os.path.join(output_dir, "generalized_solutions.csv"), index=False)
            df_best = df_resultats.sort_values(by=['CV_percent'] + optimization_targets, ascending=[False] + [False]*len(optimization_targets)).iloc[[0]]
            df_best.to_csv(os.path.join(output_dir, "best_generalized_solution.csv"), index=False)


        self.update_progress_value(100, message="Optimization completed")

        folder_results = Folder(output_dir)
        folder_results.name = "Optimization Results"

        # return the output
        return {'results_folder': folder_results}


    def optimize_model(self, name, base_model, param_dist, scorer, X, Y, kf, pipelines, n_iter=200):
        search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=n_iter,
                                    scoring='r2', cv=3, random_state=42, n_jobs=-1)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MultiOutputRegressor(search))
        ])
        score = cross_val_score(pipe, X, Y, cv=kf, scoring=scorer)
        pipelines.append((np.mean(score), pipe, name))
        return pipelines

class MilieuOptimizationProblem(ElementwiseProblem):
    def __init__(self, model, low, up, targets, all_outputs, output_thresholds, X_columns, add_cv=True, elementwise_runner=None):
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
    def __init__(self, output_dir: str, save_every=50, task=None, iterations=None):
        super().__init__()
        self.save_every = save_every
        self.history = []
        self.output_dir = output_dir
        self.task = task
        self.iterations = iterations

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        best_cv = F[:, -1].min()
        gen = algorithm.n_gen
        self.history.append({"generation": gen, "best_cv": best_cv})
        
        # Update progress during optimization
        if self.task and self.iterations:
            progress = 65 + (gen / self.iterations) * 30  # 65% to 95% for optimization
            self.task.update_progress_value(progress, message=f"Optimization generation {gen}/{self.iterations}")
        
        if gen % self.save_every == 0:
            pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "optimization_progress.csv"), index=False)