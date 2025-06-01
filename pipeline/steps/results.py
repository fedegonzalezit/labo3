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

class EvaluatePredictionsSteps(PipelineStep):

    def execute(self, df, y_test, predictions, test_index) -> None:
        
        eval_df_total = pd.DataFrame({
            "product_id": df.loc[test_index, "product_id"],
            "customer_id": df.loc[test_index, "customer_id"],
            "target": y_test["target"].values,
            "predictions": predictions.values
        })

        eval_df = eval_df_total.groupby(["product_id"]).agg({
            "target": "sum",
            "predictions": "sum"
        }).reset_index()

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


class PlotFeatureImportanceStep(PipelineStep):
    def execute(self, model) -> None:
        importance = pd.DataFrame()
        if isinstance(model, lgb.Booster):
            # Si el modelo es un Booster de LightGBM
            lgb.plot_importance(model, max_num_features=20)
            # creo el df de importancia
            importance['feature'] = model.feature_name()
            importance['importance'] = model.feature_importance(importance_type='gain')
        elif isinstance(model, XGBRegressor):
            # Si el modelo es un XGBRegressor
            xgb.plot_importance(model, max_num_features=20)
            # creo el df de importancia
            importance['feature'] = model.get_booster().feature_names
            importance['importance'] = model.get_booster().get_score(importance_type='gain').values()
        return {
            "importance": importance
        }   


class KaggleSubmissionStep(PipelineStep):
    def execute(self, df, test_index, predictions) -> None:
        submission_aux_df = pd.DataFrame({
            "product_id": df.loc[test_index, "product_id"],
            "customer_id": df.loc[test_index, "customer_id"],
            "predictions": predictions.values
        })
        submission = submission_aux_df.groupby("product_id")["predictions"].sum().reset_index()
        submission.columns = ["product_id", "tn"]
        return {"submission": submission}
    

class SaveSubmissionStep(PipelineStep):
    def __init__(self, exp_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.exp_name = exp_name

    def execute(self, submission, total_error) -> None:
        # Create the experiment directory
        exp_name = f"{str(datetime.datetime.now())}_{self.exp_name}"
        exp_dir = f"experiments/{exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        # Save the submission file
        submission.to_csv(os.path.join(exp_dir, f"submission_{self.exp_name}_{total_error}.csv"), index=False)
        

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
