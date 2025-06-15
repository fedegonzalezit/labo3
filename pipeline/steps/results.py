from pipeline import PipelineStep
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
import xgboost as xgb
import os
import pickle
from typing import Optional
import datetime
import matplotlib.pyplot as plt

class EvaluatePredictionsSteps(PipelineStep):
    def __init__(self, filter_file: Optional[str] = None, name: Optional[str] = None):
        super().__init__(name)
        self.file = filter_file

    def execute(self, df, test_index) -> None:
        
        eval_df_total = pd.DataFrame({
            "product_id": df.loc[test_index, "product_id"],
            "customer_id": df.loc[test_index, "customer_id"],
            "target": df.loc[test_index, "target"],
            "predictions": df.loc[test_index, "predictions"],
        })

        eval_df = eval_df_total.groupby(["product_id"]).agg({
            "target": "sum",
            "predictions": "sum"
        }).reset_index()
        eval_df['predictions'] = eval_df['predictions'].clip(lower=0)

        if self.file is not None:
            product_ids = pd.read_csv(self.file, sep="\t")["product_id"].tolist()
            eval_df = eval_df[eval_df["product_id"].isin(product_ids)]

        eval_df['tn_real'] = eval_df['target']
        eval_df['tn_pred'] = eval_df['predictions']

        total_error = np.sum(np.abs(eval_df['tn_real'] - eval_df['tn_pred'])) / np.sum(eval_df['tn_real'])
        print(f"Error en test: {total_error:.4f}")
        print("\nTop 5 productos con mayor error absoluto:")
        eval_df['error_absoluto'] = np.abs(eval_df['tn_real'] - eval_df['tn_pred'])
        print(eval_df.sort_values('error_absoluto', ascending=False).head())
        return {
            "eval_df": eval_df,
            "eval_df_total": eval_df_total,
            "total_error": total_error
        }

class EvaluatePredictionsOptimizatedSteps(PipelineStep):

    def execute(self, df, y_test, predictions, test_index) -> None:
        
        eval_df_total = pd.DataFrame({
            "product_id": df.loc[test_index, "product_id"],
            "customer_id": df.loc[test_index, "customer_id"],
            "target": y_test["target"].values,
            "predictions": predictions.values
        })
        # predictions clip 0 (el valor minimo de tn es 0)
        eval_df_total['predictions'] = eval_df_total['predictions'].clip(lower=0)



        # spliteo eval_df en 2 datasets al 50% de manera aleatorea para optimizacion post-train de alfa
        eval_df_total_alfa_train =  eval_df_total.sample(frac=0.5, random_state=42)
        eval_df_total_alfa_test = eval_df_total.drop(eval_df_total_alfa_train.index)
        def objective(alpha):
            df_temp = eval_df_total_alfa_train.copy()
            df_temp['predictions'] = df_temp['predictions'] * alpha
            eval_df_grouped = df_temp.groupby(["product_id"]).agg({
                "target": "sum",
                "predictions": "sum"
            }).reset_index()
            error = np.sum(np.abs(eval_df_grouped['target'] - eval_df_grouped['predictions'])) / np.sum(eval_df_grouped['target'])
            return error
        from scipy.optimize import minimize
        result = minimize(objective, x0=1.0, bounds=[(0, 10)], method='L-BFGS-B')
        alpha_opt = result.x[0]
        eval_df_total_alfa_train['predictions'] = eval_df_total_alfa_train['predictions'] * alpha_opt
        eval_df_total_alfa_test["predictions"] = eval_df_total_alfa_test['predictions'] * alpha_opt

        eval_df_train = eval_df_total_alfa_train.groupby(["product_id"]).agg({
            "target": "sum",
            "predictions": "sum"
        }).reset_index()

        eval_df_train['tn_real'] = eval_df_train['target']
        eval_df_train['tn_pred'] = eval_df_train['predictions']

        eval_df_test = eval_df_total_alfa_test.groupby(["product_id"]).agg({
            "target": "sum",
            "predictions": "sum"
        }).reset_index()
        eval_df_test["tn_real"] = eval_df_test['target']
        eval_df_test["tn_pred"] = eval_df_test['predictions']

        total_train_error = np.sum(np.abs(eval_df_train['tn_real'] - eval_df_train['tn_pred'])) / np.sum(eval_df_train['tn_real'])
        total_test_error = np.sum(np.abs(eval_df_test['tn_real'] - eval_df_test['tn_pred'])) / np.sum(eval_df_test['tn_real'])
        print(f"alfa optimizado: {alpha_opt:.4f}")
        print(f"Error en train: {total_train_error:.4f}")
        print(f"Error en test: {total_test_error:.4f}")
        print("\nTop 5 productos con mayor error absoluto en train:")
        eval_df_train['error_absoluto'] = np.abs(eval_df_train['tn_real'] - eval_df_train['tn_pred'])
        print(eval_df_train.sort_values('error_absoluto', ascending=False).head())
        print("\nTop 5 productos con mayor error absoluto en test:")
        eval_df_test['error_absoluto'] = np.abs(eval_df_test['tn_real'] - eval_df_test['tn_pred'])
        print(eval_df_test.sort_values('error_absoluto', ascending=False).head())

        return {
            "eval_df_alfa_train": eval_df_train,
            "eval_df_alfa_test": eval_df_test,
            "alpha_opt": alpha_opt
        }

class PlotFeatureImportanceStep(PipelineStep):
    def execute(self, model) -> None:
        # Plot feature importance (works for both xgboost and lightgbm wrappers)
        try:
            model.plot_importance(max_num_features=20)
        except:
            print("Could not plot feature importance for the model. It might not support this method.")

        # Si es un modelo lightgbm Booster, mostrar el DataFrame de importancias
        try:
            import lightgbm as lgb
            booster = getattr(getattr(model, "model", None), "model", None)
            if isinstance(booster, lgb.Booster):
                importance_df = pd.DataFrame({
                    "feature": booster.feature_name(),
                    "importance": booster.feature_importance(importance_type="gain")
                }).sort_values("importance", ascending=False)
                print("\nLightGBM Feature Importance (gain):")
                print(importance_df)
                return {
                    "importance_df": importance_df
                }
            elif isinstance(model, lgb.LGBMRegressor):
                importance_df = pd.DataFrame({
                    "feature": model.feature_name_,
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False)
                print("\nLightGBM Feature Importance (gain):")
                print(importance_df)
                return {
                    "importance_df": importance_df
                }
            elif isinstance(model, lgb.Booster):
                importance_df = pd.DataFrame({
                    "feature": model.feature_name(),
                    "importance": model.feature_importance(importance_type="gain")
                }).sort_values("importance", ascending=False)
                print("\nLightGBM Booster Feature Importance (gain):")
                print(importance_df)
                return {
                    "importance_df": importance_df
                }
        except Exception as e:
            print(f"Could not print LightGBM feature importance: {e}")


class KaggleSubmissionStep2(PipelineStep):
    def __init__(self, name: Optional[str] = None, filter_file: Optional[str] = None):
        super().__init__(name)
        self.file = filter_file

    def execute(self,eval_df) -> None:
        submission = pd.DataFrame({
            "product_id": eval_df["product_id"],
            "tn": eval_df["tn_pred"]
        })
        if self.file is not None:
            product_ids = pd.read_csv(self.file, sep="\t")["product_id"].tolist()
            submission = submission[submission["product_id"].isin(product_ids)]

        return {f"submission": submission}
    

class KaggleSubmissionStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, alpha_opt: float = 1, experiment="", filter_file: Optional[str] = None):
        super().__init__(name)
        self.alpha_opt = alpha_opt
        self.experiment = experiment
        self.file = filter_file


    def execute(self, df, test_index, predictions) -> None:
        submission_aux_df = pd.DataFrame({
            "product_id": df.loc[test_index, "product_id"],
            "customer_id": df.loc[test_index, "customer_id"],
            "predictions": predictions
        })
        # Apply the alpha optimization
        submission_aux_df["predictions"] *= self.alpha_opt
        submission = submission_aux_df.groupby("product_id")["predictions"].sum().reset_index()
        submission.columns = ["product_id", "tn"]
        if self.file is not None:
            product_ids = pd.read_csv(self.file, sep="\t")["product_id"].tolist()
            submission = submission[submission["product_id"].isin(product_ids)]

        return {f"submission{self.experiment}": submission}
    

class SaveSubmissionStep(PipelineStep):
    def __init__(self, exp_name: str, base_path: str, name: Optional[str] = None):
        super().__init__(name)
        self.exp_name = exp_name
        self.base_path = base_path
        self.base_path = os.path.join(self.base_path, "experiments")

    def execute(self, submission, iteration=None, total_error=None) -> None:
        # Create the experiment directory
        exp_dir = os.path.join(self.base_path, self.exp_name)
        # si ya existe el directyorio le agreo un _N al final
        #if os.path.exists(exp_dir):
        #    i = 1
        #    while os.path.exists(f"{exp_dir}_{i}"):
        #        i += 1
        #    exp_dir = f"{exp_dir}_{i}"

        os.makedirs(exp_dir, exist_ok=True)
        # Save the submission file
        base_name = f"submission_{self.exp_name}"
        if iteration is not None:
            base_name += f"_iter{iteration}"
        if total_error is not None:
            base_name += f"_{total_error:.4f}"
        submission.to_csv(os.path.join(exp_dir, f"{base_name}.csv"), index=False)
        

class SaveExperimentStep(PipelineStep):
    def __init__(self, exp_name: str, save_dataframes=False, name = None):
        super().__init__(name)
        self.exp_name = exp_name
        self.save_dataframes = save_dataframes

    def execute(self, pipeline) -> None:

        # Create the experiment directory
        exp_dir = f"experiments/{self.exp_name}"
        os.makedirs(exp_dir, exist_ok=True)

        # obtengo el model
        model = pipeline.get_artifact("model")
        # Save the model as a pickle file
        with open(os.path.join(exp_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        # guardo el error total de test
        total_error = pipeline.get_artifact("total_error")
        with open(os.path.join(exp_dir, "total_error.txt"), "w") as f:
            f.write(str(total_error))

        # Save the submission file
        submission = pipeline.get_artifact("submission")
        submission.to_csv(os.path.join(exp_dir, f"submission_{self.exp_name}_{total_error:.4f}.csv"), index=False)

        # borro submission model y error de los artifacts
        pipeline.del_artifact("submission")
        
        # Guardo los artifacts restantes que son dataframes como csvs
        if self.save_dataframes:
            for artifact_name, artifact in pipeline.artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(os.path.join(exp_dir, f"{artifact_name}.csv"), index=False)


        # Save a copy of the notebook
        #notebook_path = fallback_latest_notebook()
        #shutil.copy(notebook_path, os.path.join(exp_dir, f"notebook_{self.exp_name}.ipynb"))


class SaveDataFrameStep(PipelineStep):
    def __init__(self, df_name: str, file_name: str, ext = "pickle", name: Optional[str] = None):
        super().__init__(name)
        self.df_name = df_name
        self.file_name = file_name
        self.ext = ext

    def execute(self, pipeline) -> None:
        df = pipeline.get_artifact(self.df_name)
        if self.ext == "pickle":
            df.to_pickle(self.file_name)
        elif self.ext == "parquet":
            df.to_parquet(f"{self.file_name}.parquet", index=False)
        elif self.ext == "csv":
            df.to_csv(f"{self.file_name}.csv", index=False)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")


class SaveScalerStep(PipelineStep):
    def __init__(self, scaler_name: str, file_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.scaler_name = scaler_name
        self.file_name = file_name

    def execute(self, pipeline) -> None:
        scaler = pipeline.get_artifact(self.scaler_name)
        with open(self.file_name, "wb") as f:
            pickle.dump(scaler, f)
