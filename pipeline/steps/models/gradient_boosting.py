import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional
from pipeline import Pipeline, PipelineStep

LGB_DEFAULT_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 1024,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "n_estimators": 750,
    "verbose": -1
}

class TotalErrorMetric:
    def __init__(self, df_eval):
        self.df_eval = df_eval

    def __call__(self, preds, train_data):
        labels = train_data.get_label()
        df_temp = self.df_eval.copy()
        df_temp['preds'] = preds
        df_temp['labels'] = labels
        # Agrupar por product_id y calcular el error
        por_producto = df_temp.groupby("product_id").agg({'labels': 'sum', 'preds': 'sum'})
        # Calcular el error personalizado
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(np.abs(por_producto['labels']))
        # LightGBM espera que el segundo valor sea mayor cuando el modelo es mejor
        return 'total_error', error, False
    

class CustomMetricAutoML:
    def __init__(self, df_eval):
        self.df_eval = df_eval

    def __call__(self, X_val, y_val, estimator, *args, **kwargs):
        df_temp = X_val.copy()
        df_temp['preds'] = estimator.predict(X_val)
        df_temp['labels'] = y_val

        por_producto = df_temp.groupby("product_id").agg({'labels': 'sum', 'preds': 'sum'})
        
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(por_producto['labels'])
        
        return error, {"total_error": error}


class LGBPipelineModel:
    def __init__(self, params: Dict = LGB_DEFAULT_PARAMS):
        self.params = params
        self.model = None

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X_train, y_train, X_eval, y_eval):
        # si y_train tiene mas de una collumna uso mulltitarget
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            self.model = LGBMultiTargetPipelineModel(self.params).fit(X_train, y_train, X_eval, y_eval)
        else:
            self.model = LGBPipelineSingleTargetModel(self.params).fit(X_train, y_train, X_eval, y_eval)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

class LGBBase:
    def set_params(self, **params):
        self.params.update(params)

    def _make_datasets(self, X_train, y_train, X_eval, y_eval):
        # droppeo los indices de X_train donde y_train es nan
        y_train = y_train.dropna()
        X_train = X_train.loc[y_train.index]
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        y_eval = y_eval.dropna()
        # si y_val esta vacio, no creo el dataset de evaluacion
        if y_eval.empty:
            return train_data, None
        X_eval = X_eval.loc[y_eval.index]
        eval_data =lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features)
        return train_data, eval_data
      
    def _train_model(self, train_data, eval_data=None):
        if eval_data is None:
            eval_params = {}

        else:
            eval_params = {
                "valid_sets": [eval_data],
                "feval": TotalErrorMetric(eval_data.data)
            }
        callbacks = [
            lgb.log_evaluation(100),
            #lgb.early_stopping(50),
        ]
        model = lgb.train(
            self.params,
            train_data,
            callbacks=callbacks,
            **eval_params
        )
        return model
    

class LGBMultiTargetPipelineModel(LGBBase):
    def __init__(self, params: Dict = LGB_DEFAULT_PARAMS):
        self.params = params
        self.models = {}

    def fit(self, X_train, y_train, X_eval, y_eval):
        if not isinstance(y_train, pd.DataFrame):
            raise ValueError("y_train must be a DataFrame for multi-target regression.")
        for target in y_train.columns:
            print(f"Training model for target: {target}")
            train_data, eval_data = self._make_datasets(X_train, y_train[target], X_eval, y_eval[target])
            model = self._train_model(train_data, eval_data)
            self.models[target] = model
        return self
    
    def predict(self, X):
        if not self.models:
            raise ValueError("No models have been trained yet.")
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(X)
        return pd.DataFrame(predictions, index=X.index)


class LGBPipelineSingleTargetModel(LGBBase):
    def __init__(self, params: Dict = LGB_DEFAULT_PARAMS):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_eval, y_eval):
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            raise ValueError("y_train must be a Series for single-target regression.")
        train_data, eval_data = self._make_datasets(X_train, y_train, X_eval, y_eval)
        self.model = self._train_model(train_data, eval_data)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return pd.Series(self.model.predict(X), index=X.index, name='predictions')
    
    
class TrainModelStep(PipelineStep):
    def __init__(self, model_cls = LGBPipelineModel, name: Optional[str] = None):
        super().__init__(name)
        self.model_cls = model_cls

    def execute(self, df, X_test, y_test, X_train, y_train, params={}) -> None:
        params = params or LGB_DEFAULT_PARAMS
        model = self.model_cls(params)
        model.fit(X_train, y_train, X_test, y_test)
        return {"model": model}
    

## LEGACY
class TrainModelLGBStep(PipelineStep):
    def __init__(self, params: Dict = LGB_DEFAULT_PARAMS, train_eval_sets = {}, name: Optional[str] = None):
        super().__init__(name)
        if not params:
            params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "n_estimators": 1000,
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

    def execute(self, pipeline: Pipeline, params=None) -> None:
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])
        df_eval = pipeline.get_artifact(self.train_eval_sets["eval_data"])

        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

        
        params = params or self.params
        weight = X_train['weight'] if 'weight' in X_train.columns else None
        weight_eval = X_eval['weight'] if 'weight' in X_eval.columns else None
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, weight=weight)
        eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features, weight=weight_eval)
        custom_metric = CustomMetric(df_eval, product_id_col='product_id')
        callbacks = [
            #lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ]
        model = lgb.train(
            params,
            train_data,
            #num_boost_round=1200,
            #num_boost_round=50, # test
            valid_sets=[eval_data],
            feval=custom_metric,
            callbacks=callbacks,
        )
        return {"model": model}
    
"""

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
        
        automl = AutoML()
        metric = CustomMetricAutoML(df_eval, product_id_col='product_id')
        automl_params = {
            "time_budget": self.time_budget,
            "task": "regression",
            "metric": "rmse",
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
        automl_best_params = automl_ml_best_model.get_params()
        return {
            "automl_best_model": automl_ml_best_model,
            "automl": automl,
            "params": automl_best_params,
        }        


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

    def execute(self, pipeline: Pipeline, automl_best_model, scaler=None, scaler_target=None) -> None:
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])
        df_eval = pipeline.get_artifact(self.train_eval_sets["eval_data"])

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
        if isinstance(automl_best_model, lgb.LGBMRegressor):
            categorical_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=categorical_features)
            custom_metric = CustomMetric(df_eval, product_id_col='product_id', scaler=scaler_target)
            callbacks = [
                #lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ]
            model = lgb.train(
                automl_best_model.get_params(),
                train_data,
                num_boost_round=1000,
                valid_sets=[eval_data],
                feval=custom_metric,
                callbacks=callbacks
            )
        else:
            raise ValueError("The model is not a valid LightGBM model.")
        # Save the model
        return {"model": model}


class TrainModelXGBStep(PipelineStep):
    def __init__(self, params: Dict = {}, train_eval_sets = {}, name: Optional[str] = None):
        super().__init__(name)
        if not params:
            params = {
                'learning_rate': 0.05,
                'max_depth': 6,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 5,
                'random_state': 42,
                "enable_categorical": True,
                "verbosity": 0,
            }
        if not train_eval_sets:
            train_eval_sets = {
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
            }
        self.params = params
        self.train_eval_sets = train_eval_sets
    def execute(self, pipeline: Pipeline):
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])

        # drop rows where target is NaN
        y_train = y_train.dropna()
        X_train = X_train.loc[y_train.index]
        y_eval = y_eval.dropna()
        X_eval = X_eval.loc[y_eval.index]

        model = XGBRegressor(**self.params)
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
        return {"model": model}     
"""