from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import time
import gc
import warnings
import os
import shutil
import pickle
from flaml import AutoML
import os
from glob import glob


def fallback_latest_notebook():
    notebooks = glob("*.ipynb")
    if not notebooks:
        return None
    notebooks = sorted(notebooks, key=os.path.getmtime, reverse=True)
    return notebooks[0]



warnings.filterwarnings('ignore', category=FutureWarning)

class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    Each step in the pipeline must inherit from this class and implement the execute method.
    """
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a pipeline step.

        Args:
            name (str): Name of the step for identification and logging purposes.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def execute(self, pipeline: "Pipeline") -> None:
        """
        Execute the pipeline step.

        Args:
            pipeline (Pipeline): The pipeline instance that contains this step.
        """
        pass

    def save_artifact(self, pipeline: "Pipeline", artifact_name: str, artifact: Any) -> None:
        """
        Save an artifact produced by this step to the pipeline.

        Args:
            pipeline (Pipeline): The pipeline instance.
            artifact_name (str): Name to identify the artifact.
            artifact (Any): The artifact to save.
        """
        pipeline.save_artifact(artifact_name, artifact)


class Pipeline:
    """
    Main pipeline class that manages the execution of steps and storage of artifacts.
    """
    def __init__(self, steps: Optional[List[PipelineStep]] = None, optimize_arftifacts_memory: bool = True):
        """Initialize the pipeline."""
        self.steps: List[PipelineStep] = steps if steps is not None else []
        self.artifacts: Dict[str, Any] = {}
        self.last_step = None
        self.optimize_arftifacts_memory = optimize_arftifacts_memory

    def add_step(self, step: PipelineStep, position: Optional[int] = None) -> None:
        """
        Add a new step to the pipeline.

        Args:
            step (PipelineStep): The step to add.
            position (Optional[int]): Position where to insert the step. If None, appends to the end.
        """
        if position is not None:
            self.steps.insert(position, step)
        else:
            self.steps.append(step)

    def save_artifact(self, artifact_name: str, artifact: Any) -> None:
        """
        Save an artifact from a given step.

        Args:
            artifact_name (str): Name to identify the artifact.
            artifact (Any): The artifact to save.
        """
        if not self.optimize_arftifacts_memory:
            self.artifacts[artifact_name] = artifact
        else:
            # guarda el artifact en /tmp/ para no guardarlo en memoria
            if not os.path.exists("/tmp/"):
                os.makedirs("/tmp/")
            artifact_path = os.path.join("/tmp/", artifact_name)
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
            self.artifacts[artifact_name] = artifact_path

    def get_artifact(self, artifact_name: str) -> Any:
        """
        Retrieve a stored artifact.

        Args:
            artifact_name (str): Name of the artifact to retrieve.

        Returns:
            Any: The requested artifact.
        """
        if not self.optimize_arftifacts_memory:
            return self.artifacts.get(artifact_name)
        else:
            artifact_path = self.artifacts.get(artifact_name)
            if artifact_path and os.path.exists(artifact_path):
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
            else:
                warnings.warn(f"Artifact {artifact_name} not found in /tmp/")
                return None
    
    def del_artifact(self, artifact_name: str, soft=True) -> None:
        """
        Delete a stored artifact and free memory.

        Args:
            artifact_name (str): Name of the artifact to delete.
        """
        del self.artifacts[artifact_name]
        if not soft:
            # Force garbage collection if not soft delete
            gc.collect()
    
    def before_step_callback(self) -> None:
        """
        Set a callback to be called before each step execution.
        """
        pass
        
    def after_step_callback(self) -> None:
        """
        Set a callback to be called after each step execution.
        """
        pass
        
    def after_last_step_callback(self) -> None:
        """
        Set a callback to be called after the last step execution.
        """
        pass
    
    def run(self, verbose: bool = True, last_step_callback: Callable = None) -> None:
        """
        Execute all steps in sequence and log execution time.
        """        
        
        # Run steps from the last completed step
        for step in self.steps:
            if verbose:
                print(f"Executing step: {step.name}")
            start_time = time.time()
            self.before_step_callback() 
            step.execute(self)
            self.after_step_callback()
            end_time = time.time()
            if verbose:
                print(f"Step {step.name} completed in {end_time - start_time:.2f} seconds")
            self.last_step = step
            if step == self.steps[-1]:
                self.after_last_step_callback()
                if last_step_callback:
                    last_step_callback(self)   

    def clear(self, collect_garbage: bool = False) -> None:
        """
        Clean up all artifacts and free memory.
        """
        if collect_garbage:
            del self.artifacts
            gc.collect()
        self.artifacts = {}
        self.last_step = None


import pandas as pd
import numpy as np
import lightgbm as lgb


class LoadDataFrameStep(PipelineStep):
    """
    Example step that loads a DataFrame.
    """
    def __init__(self, path: str, name: Optional[str] = None):
        super().__init__(name)
        self.path = path

    def execute(self, pipeline: Pipeline) -> None:
        df = pd.read_parquet(self.path)
        df = df.drop(columns=["periodo"])
        self.save_artifact(pipeline, "df", df)


class CastDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for col, dtype in self.dtypes.items():
            df[col] = df[col].astype(dtype)
        print(df.info())
        self.save_artifact(pipeline, "df", df)


class ChangeDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for original_dtype, dtype in self.dtypes.items():
            for col in df.select_dtypes(include=[original_dtype]).columns:
                df[col] = df[col].astype(dtype)
        print(df.info())
        self.save_artifact(pipeline, "df", df)


class FilterFirstDateStep(PipelineStep):
    def __init__(self, first_date: str, name: Optional[str] = None):
        super().__init__(name)
        self.first_date = first_date

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        df = df[df["fecha"] >= self.first_date]
        print(f"Filtered DataFrame shape: {df.shape}")
        self.save_artifact(pipeline, "df", df)


class FeatureEngineeringLagStep(PipelineStep):
    def __init__(self, lags: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.lags = lags
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for col in self.columns:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] =  df.groupby(['product_id', 'customer_id'])[col].shift(lag)
        self.save_artifact(pipeline, "df", df)


import tqdm
class FeatureEngineeringProductInteractionStep(PipelineStep):

    def execute(self, pipeline: Pipeline) -> None:
        """
        El dataframe tiene una columna product_id y customer_id y fecha.
        Quiero obtener los 100 productos con mas tn del ultimo mes y crear 100 nuevas columnas que es la suma de tn de esos productos (para todos los customer)
        se deben agregan entonces respetando la temporalidad la columna product_{product_id}_total_tn
        """
        df = pipeline.get_artifact("df")
        last_date = df["fecha"].max()
        last_month_df = df[df["fecha"] == last_date]
        top_products = last_month_df.groupby("product_id").aggregate({"tn": "sum"}).nlargest(10, "tn").index.tolist()
        # TODO: mejor agruparlo por categoria y hacer una columna por cada categoria tanto de agrup por product como por customer
        for product_id in tqdm.tqdm(top_products):
            # creo un subset que es el total de product_id vendidos para todos los customer en cada t y lo mergeo a df
            product_df = df[df["product_id"] == product_id].groupby("fecha").aggregate({"tn": "sum"}).reset_index()
            product_df = product_df.rename(columns={"tn": f"product_{product_id}_total_tn"})
            product_df = product_df[["fecha", f"product_{product_id}_total_tn"]]
            df = df.merge(product_df, on="fecha", how="left")
        self.save_artifact(pipeline, "df", df)


class FeatureEngineeringProductCatInteractionStep(PipelineStep):

    def __init__(self, cat="cat1", name: Optional[str] = None):
        super().__init__(name)
        self.cat = cat


    def execute(self, pipeline: Pipeline) -> None:
        # agrupo el dataframe por cat1 (sumando), obteniendo fecha, cat1 y
        # luego paso el dataframe a wide format, donde cada columna es una categoria  y la fila es la suma de tn para cada cat1
        # luego mergeo al dataframe original por fecha y product_id
        df = pipeline.get_artifact("df")
        df_cat = df.groupby(["fecha", self.cat]).agg({"tn": "sum"}).reset_index()
        df_cat = df_cat.pivot(index="fecha", columns=self.cat, values="tn").reset_index()
        df = df.merge(df_cat, on="fecha", how="left")
        self.save_artifact(pipeline, "df", df)
        

class DateRelatedFeaturesStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        df["year"] = df["fecha"].dt.year
        df["mes"] = df["fecha"].dt.month
        self.save_artifact(pipeline, "df", df)

        
class SplitDataFrameStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        sorted_dated = sorted(df["fecha"].unique())
        last_date = sorted_dated[-1] # es 12-2019
        last_test_date = sorted_dated[-3] # needs a gap because forecast moth+2
        last_train_date = sorted_dated[-4] #

        kaggle_pred = df[df["fecha"] == last_date]
        test = df[df["fecha"] == last_test_date]
        eval_data = df[df["fecha"] == last_train_date]
        train = df[(df["fecha"] < last_train_date)]
        self.save_artifact(pipeline, "train", train)
        self.save_artifact(pipeline, "eval_data", eval_data)
        self.save_artifact(pipeline, "test", test)
        self.save_artifact(pipeline, "kaggle_pred", kaggle_pred)


class CustomMetric:
    def __init__(self, df_eval, product_id_col='product_id', scaler=None):
        self.scaler = scaler
        self.df_eval = df_eval
        self.product_id_col = product_id_col
    
    def __call__(self, preds, train_data):
        labels = train_data.get_label()
        df_temp = self.df_eval.copy()
        df_temp['preds'] = preds
        df_temp['labels'] = labels

        if self.scaler:
            df_temp['preds'] = self.scaler.inverse_transform(df_temp[['preds']])
            df_temp['labels'] = self.scaler.inverse_transform(df_temp[['labels']])
        
        # Agrupar por product_id y calcular el error
        por_producto = df_temp.groupby(self.product_id_col).agg({'labels': 'sum', 'preds': 'sum'})
        
        # Calcular el error personalizado
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(por_producto['labels'])
        
        # LightGBM espera que el segundo valor sea mayor cuando el modelo es mejor
        return 'custom_error', error, False


    
class CustomMetricAutoML:
    def __init__(self, df_eval, product_id_col='product_id', scaler=None):
        self.df_eval = df_eval
        self.product_id_col = product_id_col
        self.scaler = scaler

    def __call__(self, X_val, y_val, estimator, *args, **kwargs):
        df_temp = X_val.copy()
        df_temp['preds'] = estimator.predict(X_val)
        df_temp['labels'] = y_val

        if self.scaler:
            df_temp['preds'] = self.scaler.inverse_transform(df_temp[['preds']])
            df_temp['labels'] = self.scaler.inverse_transform(df_temp[['labels']])
        
        # Agrupar por product_id y calcular el error
        por_producto = df_temp.groupby(self.product_id_col).agg({'labels': 'sum', 'preds': 'sum'})
        
        # Calcular el error personalizado
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(por_producto['labels'])
        
        return error, {"total_error": error}

from sklearn.preprocessing import RobustScaler, MinMaxScaler

class TrainScalerDataStep(PipelineStep):
    def __init__(self, scaler = RobustScaler, name: Optional[str] = None):
        super().__init__(name)
        self.scaler = scaler

    def execute(self, pipeline: Pipeline) -> None:
        train = pipeline.get_artifact("train")        
        scaler = self.scaler()

        # escalo las columnas que son int32 o float32
        columns_to_scale = train.select_dtypes(include=['int32', 'float32']).columns.tolist()
        # saco la columna target
        columns_to_scale = [col for col in columns_to_scale if col not in
                ["periodo", 'fecha', 'target']]

        scaler = scaler.fit(train[columns_to_scale])
        scaler_target = self.scaler()
        scaler_target = scaler_target.fit(train[["target"]])

        self.save_artifact(pipeline, "scaler", scaler)
        self.save_artifact(pipeline, "scaler_target", scaler_target)
        self.save_artifact(pipeline, "columns_to_scale", columns_to_scale)
        

class PrepareXYStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        train = pipeline.get_artifact("train")
        eval_data = pipeline.get_artifact("eval_data")
        test = pipeline.get_artifact("test")
        kaggle_pred = pipeline.get_artifact("kaggle_pred")

        features = [col for col in train.columns if col not in
                        ['fecha', 'target']]
        target = 'target'

        X_train = pd.concat([train[features], eval_data[features]]) # [train + eval] + [eval] -> [test] 
        y_train = pd.concat([train[target], eval_data[target]])

        X_train_alone = train[features]
        y_train_alone = train[target]

        X_eval = eval_data[features]
        y_eval = eval_data[target]

        X_test = test[features]
        y_test = test[target]

        X_train_final = pd.concat([train[features], eval_data[features], test[features]])
        y_train_final = pd.concat([train[target], eval_data[target], test[target]])

        X_kaggle = kaggle_pred[features]
        self.save_artifact(pipeline, "X_train", X_train)
        self.save_artifact(pipeline, "y_train", y_train)
        self.save_artifact(pipeline, "X_train_alone", X_train_alone)
        self.save_artifact(pipeline, "y_train_alone", y_train_alone)
        self.save_artifact(pipeline, "X_eval", X_eval)
        self.save_artifact(pipeline, "y_eval", y_eval)
        self.save_artifact(pipeline, "X_test", X_test)
        self.save_artifact(pipeline, "y_test", y_test)
        self.save_artifact(pipeline, "X_train_final", X_train_final)
        self.save_artifact(pipeline, "y_train_final", y_train_final)
        self.save_artifact(pipeline, "X_kaggle", X_kaggle)
        

class TrainModelLGBStep(PipelineStep):
    def __init__(self, params: Dict = {}, train_eval_sets = {}, name: Optional[str] = None):
        super().__init__(name)
        if not params:
            params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1
            }
        if not train_eval_sets:
            train_eval_sets = {
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data",
            }
        self.params = params
        self.train_eval_sets = train_eval_sets

    def execute(self, pipeline: Pipeline) -> None:
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])
        df_eval = pipeline.get_artifact(self.train_eval_sets["eval_data"])

        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

        scaler = pipeline.get_artifact("scaler")
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler:
            X_train[scaler.feature_names_in_] = scaler.transform(X_train[scaler.feature_names_in_])
            X_eval[scaler.feature_names_in_] = scaler.transform(X_eval[scaler.feature_names_in_])
            y_train = pd.Series(
                scaler_target.transform(y_train.values.reshape(-1, 1)).flatten(),
                index=y_train.index,
            )
            y_eval = pd.Series(
                scaler_target.transform(y_eval.values.reshape(-1, 1)).flatten(),
                index=y_eval.index,
            )

        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features)
        custom_metric = CustomMetric(df_eval, product_id_col='product_id', scaler=scaler_target)
        callbacks = [
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ]
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            #num_boost_round=50, # test
            valid_sets=[eval_data],
            feval=custom_metric,
            callbacks=callbacks
        )
        # Save the model
        self.save_artifact(pipeline, "model", model)


class TrainModelAutoMLStep(PipelineStep):
    def __init__(self, train_eval_sets = {}, time_budget: int = 100, products_proportion: float = 1.0, name: Optional[str] = None):
        super().__init__(name)
        self.time_budget = time_budget
        if not train_eval_sets:
            train_eval_sets = {
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data",
            }
        self.train_eval_sets = train_eval_sets
        self.products_proportion = products_proportion

    def execute(self, pipeline: Pipeline) -> None:
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])
        df_eval = pipeline.get_artifact(self.train_eval_sets["eval_data"])

        # para que sea mas rapido si self.products_proportion < 1.0, tomo una muestra de los productos
        if self.products_proportion < 1.0:
            unique_products = X_train['product_id'].unique()
            sample_size = int(len(unique_products) * self.products_proportion)
            sampled_products = np.random.choice(unique_products, size=sample_size, replace=False)
            X_train = X_train[X_train['product_id'].isin(sampled_products)]
            y_train = y_train[X_train.index]
            X_eval = X_eval[X_eval['product_id'].isin(sampled_products)]
            y_eval = y_eval[X_eval.index]
            df_eval = df_eval[df_eval['product_id'].isin(sampled_products)]
        
        scaler = pipeline.get_artifact("scaler")
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler:
            X_train[scaler.feature_names_in_] = scaler.transform(X_train[scaler.feature_names_in_])
            X_eval[scaler.feature_names_in_] = scaler.transform(X_eval[scaler.feature_names_in_])
            y_train = pd.Series(
                scaler_target.transform(y_train.values.reshape(-1, 1)).flatten(),
                index=y_train.index,
            )
            y_eval = pd.Series(
                scaler_target.transform(y_eval.values.reshape(-1, 1)).flatten(),
                index=y_eval.index,
            )
        automl = AutoML()
        metric = CustomMetricAutoML(df_eval, product_id_col='product_id', scaler=scaler_target)
        automl_params = {
            "time_budget": self.time_budget,
            "task": "regression",
            "metric": metric,
            "eval_method": "holdout",
            #"estimator_list": ["lgbm", "xgboost", "catboost"],
            "estimator_list": ["lgbm"],
        }
            
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_eval,
            y_val=y_eval,
            **automl_params
        )
        # Save the model
        automl_ml_best_model = automl.model.estimator
        self.save_artifact(pipeline, "model", automl)
        
        self.save_artifact(pipeline, "automl_ml_best_model", automl_ml_best_model)


class ReTrainAutoMLBestModelStep(PipelineStep):
    def __init__(self, train_eval_sets = {}, name: Optional[str] = None):
        super().__init__(name)
        if not train_eval_sets:
            train_eval_sets = {
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data",
            }
        self.train_eval_sets = train_eval_sets

    def execute(self, pipeline: Pipeline) -> None:
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])
        df_eval = pipeline.get_artifact(self.train_eval_sets["eval_data"])

        scaler = pipeline.get_artifact("scaler")
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler:
            X_train[scaler.feature_names_in_] = scaler.transform(X_train[scaler.feature_names_in_])
            X_eval[scaler.feature_names_in_] = scaler.transform(X_eval[scaler.feature_names_in_])
            y_train = pd.Series(
                scaler_target.transform(y_train.values.reshape(-1, 1)).flatten(),
                index=y_train.index,
            )
            y_eval = pd.Series(
                scaler_target.transform(y_eval.values.reshape(-1, 1)).flatten(),
                index=y_eval.index,
            )
        automl = pipeline.get_artifact("automl_ml_best_model")
        if isinstance(automl, lgb.LGBMRegressor):
            categorical_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=categorical_features)
            custom_metric = CustomMetric(df_eval, product_id_col='product_id', scaler=scaler_target)
            callbacks = [
                #lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ]
            model = lgb.train(
                automl.get_params(),
                train_data,
                num_boost_round=1000,
                valid_sets=[eval_data],
                feval=custom_metric,
                callbacks=callbacks
            )
        else:
            raise ValueError("The model is not a valid LightGBM model.")
        # Save the model
        self.save_artifact(pipeline, "model", model)


class PredictStep(PipelineStep):
    def __init__(self, predict_set: str, name: Optional[str] = None):
        super().__init__(name)
        self.predict_set = predict_set

    def execute(self, pipeline: Pipeline) -> None:
        X_predict = pipeline.get_artifact(self.predict_set)
        scaler = pipeline.get_artifact("scaler")
        if scaler:
            X_predict[scaler.feature_names_in_] = scaler.transform(X_predict[scaler.feature_names_in_])
        model = pipeline.get_artifact("model")
        predictions = model.predict(X_predict)
        # los valores de predictions que dan menores a 0 los seteo en 0
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler_target:
            predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
        # la columna de predictions seria "predictions" y le agrego columna de product_id
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=X_predict.index)
        predictions["product_id"] = X_predict["product_id"]
        self.save_artifact(pipeline, "predictions", predictions)


class EvaluatePredictionsSteps(PipelineStep):
    def __init__(self, y_actual_df: str, name: Optional[str] = None):
        super().__init__(name)
        self.y_actual_df = y_actual_df

    def execute(self, pipeline: Pipeline) -> None:
        predictions = pipeline.get_artifact("predictions")
        y_actual = pipeline.get_artifact(self.y_actual_df)

        product_actual = y_actual.groupby("product_id")["target"].sum()
        product_pred = predictions.groupby("product_id")["predictions"].sum()

        eval_df = pd.DataFrame({
            "product_id": product_actual.index,
            "tn_real": product_actual.values,
            "tn_pred": product_pred.values
        })

        total_error = np.sum(np.abs(eval_df['tn_real'] - eval_df['tn_pred'])) / np.sum(eval_df['tn_real'])
        print(f"Error en test: {total_error:.4f}")
        print("\nTop 5 productos con mayor error absoluto:")
        eval_df['error_absoluto'] = np.abs(eval_df['tn_real'] - eval_df['tn_pred'])
        print(eval_df.sort_values('error_absoluto', ascending=False).head())
        self.save_artifact(pipeline, "eval_df", eval_df)
        self.save_artifact(pipeline, "total_error", total_error)


class PlotFeatureImportanceStep(PipelineStep):

    def execute(self, pipeline: Pipeline) -> None:
        model = pipeline.get_artifact("model")
        lgb.plot_importance(model)


class KaggleSubmissionStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        model = pipeline.get_artifact("model")
        kaggle_pred = pipeline.get_artifact("kaggle_pred")
        X_kaggle = pipeline.get_artifact("X_kaggle")
        scaler = pipeline.get_artifact("scaler")
        if scaler:
            X_kaggle[scaler.feature_names_in_] = scaler.transform(X_kaggle[scaler.feature_names_in_])
        preds = model.predict(X_kaggle)
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler_target:
            preds = scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
        #kaggle_pred["tn_predicha"] = model.predict(X_kaggle) # try using .loc[row_indexer, col_indexer] = value instead
        kaggle_pred["tn_predicha"] = preds.copy()
        submission = kaggle_pred.groupby("product_id")["tn_predicha"].sum().reset_index()
        submission.columns = ["product_id", "tn"]
        self.save_artifact(pipeline, "submission", submission)


class SaveExperimentStep(PipelineStep):
    def __init__(self, exp_name: str, save_dataframes=False, name: Optional[str] = None):
        super().__init__(name)
        self.exp_name = exp_name
        self.save_dataframes = save_dataframes

    def execute(self, pipeline: Pipeline) -> None:

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
        pipeline.del_artifact("model")
        pipeline.del_artifact("total_error")
        
        # Guardo los artifacts restantes que son dataframes como csvs
        if self.save_dataframes:
            for artifact_name, artifact in pipeline.artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(os.path.join(exp_dir, f"{artifact_name}.csv"), index=False)


        # Save a copy of the notebook
        notebook_path = fallback_latest_notebook()
        shutil.copy(notebook_path, os.path.join(exp_dir, f"notebook_{self.exp_name}.ipynb"))


import datetime

pipeline = Pipeline(
    steps=[
        LoadDataFrameStep(path="df_intermedio.parquet"),
        FeatureEngineeringLagStep(lags=[1,2,3,6,12], columns=["tn", "cust_request_qty", "stock_final"]),
        DateRelatedFeaturesStep(),
        #FeatureEngineeringProductInteractionStep(),
        FeatureEngineeringProductCatInteractionStep(cat="cat1"),
        FeatureEngineeringProductCatInteractionStep(cat="cat2"),
        #FeatureEngineeringProductCatInteractionStep(cat="cat3"),
        CastDataTypesStep(dtypes=
            {
                "product_id": "uint32", 
                "customer_id": "uint32",
                "mes": "uint16",
                "year": "uint16",
            }
        ),
        ChangeDataTypesStep(dtypes={
            "float64": "float32",
        }),
        FilterFirstDateStep(first_date="2018-01-01"), # para que sea mas liviano el dataset y poder hacer mas pruebas
        SplitDataFrameStep(),
        #TrainScalerDataStep(scaler=RobustScaler),
        PrepareXYStep(),
        TrainModelAutoMLStep(
            train_eval_sets={
                "X_train": "X_train_alone",
                "y_train": "y_train_alone",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data"
            },
            time_budget=int(60*2),
            products_proportion=1
        ),
        ReTrainAutoMLBestModelStep(
            train_eval_sets={
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data"
            },
        ),
        PredictStep(predict_set="X_test"),
        EvaluatePredictionsSteps(y_actual_df="test"),
        PlotFeatureImportanceStep(),
        ReTrainAutoMLBestModelStep(
            train_eval_sets={
                "X_train": "X_train_final",
                "y_train": "y_train_final",
                "X_eval": "X_test",
                "y_eval": "y_test",
                "eval_data": "test",
            },
        ),
        KaggleSubmissionStep(),
        SaveExperimentStep(exp_name=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_exp_automl", save_dataframes=False),
    ],
    optimize_arftifacts_memory=True
)
pipeline.run(verbose=True)