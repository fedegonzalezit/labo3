import numpy as np
import lightgbm as lgb
from typing import Dict, Optional
from pipeline import Pipeline, PipelineStep

LGB_DEFAULT_PARAMS = {
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
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(por_producto['labels'])
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




class TrainModelLGBStep(PipelineStep):
    def __init__(self, params: Dict = {}, train_eval_sets = {}, name: Optional[str] = None):
        super().__init__(name)
        if not train_eval_sets:
            train_eval_sets = {
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data",
            }
        self.params = params or LGB_DEFAULT_PARAMS
        self.train_eval_sets = train_eval_sets

    def execute(self, pipeline: Pipeline, params=None) -> None:


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