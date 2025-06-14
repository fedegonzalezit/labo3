import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional
from pipeline import Pipeline, PipelineStep
from sklearn.neural_network import MLPRegressor

LGB_DEFAULT_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "n_estimators": 1500,
    # bins 1024
    "max_bin": 1024,
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

from sklearn.model_selection import KFold, cross_val_score
# import deepclone
from copy import deepcopy

class EnsambleKFoldWrapper:
    # esta funcion se inicializa con un modelo, los clona y entra N modelos, uno por cada kfold.
    # cuando hace la prediccion, promedia las predicciones de los N modelos.
    def __init__(self, model, n_splits=5):
        self.model = model
        self.n_splits = n_splits
        self.models = []

    def fit(self, X, y, X_val, y_val):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(X):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            model_clone = deepcopy(self.model)  # Clona el modelo para cada fold
            model_clone.fit(X_train, y_train, X_val, y_val)
            self.models.append(model_clone)
        return self
    def predict(self, X):
        if not self.models:
            raise ValueError("No models have been trained yet.")
        predictions = np.mean([model.predict(X) for model in self.models], axis=0)
        return pd.Series(predictions, index=X.index, name='predictions')


class XGBOOSTPipelineModel:
    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.model = None

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X_train, y_train, X_eval, y_eval, weight=None):
        from xgboost import XGBRegressor
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            raise ValueError("y_train must be a Series for single-target regression.")
        self.model = XGBRegressor(**self.params, enable_categorical=True)
        eval_sets = []
        y_eval = y_eval.dropna()
        if not y_eval.empty:
            X_eval = X_eval.loc[y_eval.index]
            for column in X_eval.columns:
                # replace inf and -inf with 0
                X_eval[column] = X_eval[column].replace([np.inf, -np.inf], 0)

            eval_sets = [(X_eval, y_eval)]
        y_train = y_train.dropna()
        X_train = X_train.loc[y_train.index]
        for column in X_train.columns:
            # replace inf and -inf with 0
            X_train[column] = X_train[column].replace([np.inf, -np.inf], 0)
        self.model.fit(X_train, y_train, eval_set=eval_sets, verbose=True)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        for column in X.columns:
            # replace inf and -inf with 0
            X[column] = X[column].replace([np.inf, -np.inf], 0)
        return pd.Series(self.model.predict(X), index=X.index, name='predictions')

from sklearn.linear_model import Ridge

class RidgePipelineModel:
    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.model = Ridge(**self.params)
        self.features = []

    def set_params(self, **params):
        self.params.update(params)
        self.model.set_params(**self.params)

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, weight=None):
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            raise ValueError("y_train must be a Series for single-target regression.")
        y_train = y_train.dropna()
        X_train = X_train.loc[y_train.index]
        # elimino todas las columnas que no son numéricas
        X_train = X_train.select_dtypes(include=[np.number])
        # fillna con 0
        X_train = X_train.fillna(0)
        self.features = X_train.columns.tolist()
        for column in X_train.columns:
            # replace inf and -inf with 0
            X_train[column] = X_train[column].replace([np.inf, -np.inf], 0)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X = X[self.features]  # Asegurarse de que X tenga las mismas columnas que se usaron para entrenar
        X = X.fillna(0)  # Rellenar NaN con 0
        for column in X.columns:
            # replace inf and -inf with 0
            X[column] = X[column].replace([np.inf, -np.inf], 0)

        return pd.Series(self.model.predict(X), index=X.index, name='predictions')
    

class LGBPipelineModel:
    def __init__(self, params: Dict = LGB_DEFAULT_PARAMS, callbacks=[]):
        self.params = params
        self.model = None
        self.callbacks = callbacks  # Callbacks to be used during training, if applicable


    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X_train, y_train, X_eval, y_eval, weight=None):
        # si y_train tiene mas de una collumna uso mulltitarget
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            self.model = LGBMultiTargetPipelineModel(self.params, self.callbacks).fit(X_train, y_train, X_eval, y_eval, weight=weight)
        else:
            self.model = LGBPipelineSingleTargetModel(self.params, self.callbacks).fit(X_train, y_train, X_eval, y_eval, weight=weight)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def plot_importance(self, max_num_features=20):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        lgb.plot_importance(self.model.model, max_num_features=max_num_features)

class LGBBase:
    def set_params(self, **params):
        self.params.update(params)

    def _make_datasets(self, X_train, y_train, X_eval, y_eval, weight=None):
        # droppeo los indices de X_train donde y_train es nan
        y_train = y_train.dropna()
        X_train = X_train.loc[y_train.index]
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
        if weight is not None:
            train_weight = weight.loc[y_train.index]
        else:
            train_weight = None
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, weight=train_weight)
        y_eval = y_eval.dropna()
        # si y_val esta vacio, no creo el dataset de evaluacion
        print(f"Validation set size: {len(y_eval)}")
        if y_eval.empty:
            return train_data, None
        del X_train, y_train  # borro X_train, y_train para liberar memoria
        X_eval = X_eval.loc[y_eval.index]
        print(f"X_eval first 5 rows:\n{X_eval.head()}")
        print(f"y_eval first 5 rows:\n{y_eval.head()}")
        if weight is not None:
            eval_weight = weight.loc[y_eval.index]
        else:
            eval_weight = None
        eval_data =lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features, weight=eval_weight)
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
            *self.callbacks
        ]
        print(callbacks)
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

    def fit(self, X_train, y_train, X_eval, y_eval, weight=None):
        if not isinstance(y_train, pd.DataFrame):
            raise ValueError("y_train must be a DataFrame for multi-target regression.")
        for target in y_train.columns:
            print(f"Training model for target: {target}")
            train_data, eval_data = self._make_datasets(X_train, y_train[target], X_eval, y_eval[target], weight=weight)
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
    def __init__(self, params: Dict = LGB_DEFAULT_PARAMS, callbacks=[]):
        self.params = params
        self.model = None
        self.callbacks = callbacks 

    def fit(self, X_train, y_train, X_eval, y_eval, weight=None):
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            raise ValueError("y_train must be a Series for single-target regression.")
        train_data, eval_data = self._make_datasets(X_train, y_train, X_eval, y_eval, weight=weight)
        # borro X_train, y _train, X_eval, y_eval para liberar memoria
        del X_train, y_train, X_eval, y_eval
        self.model = self._train_model(train_data, eval_data)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return pd.Series(self.model.predict(X), index=X.index, name='predictions')
    
    
class TrainModelStep(PipelineStep):
    def __init__(self, model_cls = LGBPipelineModel, name: Optional[str] = None, params={}, folds=0, callbacks=[]):
        super().__init__(name)
        self.model_cls = model_cls
        self.params = params
        self.folds = folds  # Number of folds for cross-validation, if applicable
        self.callbacks = callbacks 

    def execute(self, pipeline, X_test, y_test, X_train, y_train, weights=None, params={}) -> None:
        # instancio los callbacks
        self.callbacks = [callback(pipeline) for callback in self.callbacks]
        params = params or self.params
        if self.folds > 1:
            model = self.model_cls(callbacks=self.callbacks)
            model.set_params(**params)
            model = EnsambleKFoldWrapper(model, n_splits=self.folds)

            model.fit(X_train, y_train, X_test, y_test, weight=weights)
        else:
            model =  self.model_cls(callbacks=self.callbacks)
            model.set_params(**params)
            model.fit(X_train, y_train, X_test, y_test, weight=weights)
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
    

class MLPPipelineModel:
    def __init__(self, params: Dict = None):
        self.params = params or {
            "hidden_layer_sizes": (256, 128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "random_state": 42,
            "verbose": True,
        }
        self.model = None
        self.feature_columns = None  # Para guardar las columnas después del one-hot

    def set_params(self, **params):
        self.params.update(params)

    def _preprocess(self, X):
        # Convierte variables categóricas a dummies
        X_proc = pd.get_dummies(X, drop_first=True)
        # Si ya entrenamos, aseguramos que las columnas coincidan
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in X_proc:
                    X_proc[col] = 0
            X_proc = X_proc[self.feature_columns]
        # Reemplaza NaN por 0
        X_proc = X_proc.fillna(0)
        return X_proc

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            raise ValueError("y_train debe ser una Serie para regresión single-target.")
        y_train = y_train.dropna()
        X_train = X_train.loc[y_train.index]
        X_train_proc = pd.get_dummies(X_train, drop_first=True)
        X_train_proc = X_train_proc.fillna(0)  # <--- asegurate de esto
        self.feature_columns = X_train_proc.columns.tolist()
        self.model = MLPRegressor(**self.params)
        self.model.fit(X_train_proc, y_train)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been entrenado aún.")
        X_proc = self._preprocess(X)
        return pd.Series(self.model.predict(X_proc), index=X.index, name='predictions')