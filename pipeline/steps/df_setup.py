from typing import Optional, Dict
from pipeline import PipelineStep
import pandas as pd

class SplitDataFrameStep(PipelineStep):
    def __init__(
            self, 
            test_date="2019-12", 
            df="df", 
            gap=0,
            name: Optional[str] = None
        ):
        super().__init__(name)
        self.test_date = test_date
        self.df = df
        self.gap = gap 

    def execute(self, pipeline) -> None:
        df = pipeline.get_artifact(self.df)
        test_df = df[df["fecha"] == self.test_date]
        train_df = df[df["fecha"] < self.test_date]
        last_train_date = train_df["fecha"].max()
        if isinstance(last_train_date, pd.Period):
            last_train_date = last_train_date.to_timestamp()
        gap_date = pd.to_datetime(last_train_date) - pd.DateOffset(months=self.gap)
        # Convert gap_date to Period with same freq as fecha
        if pd.api.types.is_period_dtype(df["fecha"]):
            gap_date = pd.Period(gap_date, freq=df["fecha"].dt.freq)
        train_df = train_df[train_df["fecha"] < gap_date]
        return {
            "train_index": train_df.index,
            "test_index": test_df.index
        }


class PrepareXYStep(PipelineStep):
    def execute(self, df, train_index, test_index) -> None:
        columns = df.columns
        #features = [col for col in columns if col != "fecha" and "target" not in col]
        features = [col for col in columns if col != "fecha" and "target" not in col]
        targets = [col for col in columns if "target" in col]
        X_train = df.loc[train_index][features]
        y_train = df.loc[train_index][targets]
        X_test = df.loc[test_index][features]
        y_test = df.loc[test_index][targets]
        return {
            "features": features,
            "targets": targets,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }


class CreateTargetColumStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, target_col: str = 'tn'):
        super().__init__(name)
        self.target_col = target_col

    def execute(self, df: pd.DataFrame) -> Dict:

        df = df.sort_values(['product_id', 'customer_id', 'fecha'])
        df['target'] = df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-2)    
        return {"df": df, "target_col": self.target_col}
    

class CreateMultiDiffTargetColumStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, target_col: str = 'tn'):
        super().__init__(name)
        self.target_col = target_col

    def execute(self, df: pd.DataFrame) -> Dict:

        df = df.sort_values(['product_id', 'customer_id', 'fecha'])
        df['target_1'] = df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-1) - df[self.target_col]
        df['target_2'] = df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-2) - df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-1)
        return {
            "df": df, 
            "target_col": self.target_col,
            "needs_integration": True,
            #"integration_function": lambda x: x[self.target] + x['target_1'] + x['target_2']
        }


class CreateTargetColumDiffStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, target_col: str = 'tn'):
        super().__init__(name)
        self.target_col = target_col

    def execute(self, df: pd.DataFrame) -> Dict:
        df.drop(columns=["target"], inplace=True, errors='ignore')
        df = df.sort_values(['product_id', 'customer_id', 'fecha'])
        df['target'] = df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-2) - df[self.target_col]
        return {
            "df": df, 
            "target_col": self.target_col,
            "needs_integration": True,
            #"integration_function": lambda x: x[self.target] + x['target']
        }
    

class PredictStep(PipelineStep):
    def execute(self, df, test_index, model, features) -> None:
        X_predict = df.loc[test_index][features]
        predictions = model.predict(X_predict)
        return {"predictions": predictions}


class IntegratePredictionsStep(PipelineStep):
    def execute(self, df, predictions, test_index, target_col, needs_integration=False) -> Dict:
        if not needs_integration:
            return {
                "y_test": df.loc[test_index, ["target"]]
            }
        # crea un nuevo dataframe que es la suma de todas las columnas de predicciones
        if predictions.ndim == 1:
            predictions_sum = pd.Series(predictions, index=test_index, name='predictions')
        else:
            predictions_sum = predictions.sum(axis=1)
        final_predictions = predictions_sum + df.loc[test_index, target_col]
        predictions = pd.Series(final_predictions, index=test_index, name='predictions')
        target_columns = [col for col in df.columns if 'target' in col]
        test_sum = df.loc[test_index, target_columns].sum(axis=1)
        y_test = test_sum + df.loc[test_index, target_col]
        y_test = pd.DataFrame(y_test, index=test_index, columns=["target"])
        
        # nuevo approach, uso integration_function
        
        
        return {
            "predictions": predictions,
            "y_test": y_test
        }
    

## legacy code
class IntegratePredictionsStepOld(PipelineStep):
    def execute(self, pipeline, predict_set, predictions, target_col, test) -> Dict:
        """
        Integra las predicciones al DataFrame de test.
        Si el target_col es una diferencia, se suma el último valor de target_col al target.
        """
        pred_original_df = pipeline.get_artifact(predict_set)
        predictions["predictions"] = predictions["predictions"] + pred_original_df[target_col]
        test["target"] = test["target"] + test[target_col]
        return {
            "predictions": predictions,
            "test": test
        } 



class SplitDataFrameStepOld(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, df) -> None:
        sorted_dated = sorted(df["fecha"].unique())
        last_date = sorted_dated[-1] # es 12-2019
        last_test_date = sorted_dated[-3] # needs a gap because forecast moth+2
        last_train_date = sorted_dated[-4] #

        kaggle_pred = df[df["fecha"] == last_date]
        test = df[df["fecha"] == last_test_date]
        eval_data = df[df["fecha"] == last_train_date]
        train = df[(df["fecha"] < last_train_date)]
        return {
            "train": train,
            "eval_data": eval_data,
            "test": test,
            "kaggle_pred": kaggle_pred
        }
    

class PrepareXYStepOld(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, train, eval_data, test, kaggle_pred) -> None:
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
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_train_alone": X_train_alone,
            "y_train_alone": y_train_alone,
            "X_eval": X_eval,
            "y_eval": y_eval,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_final": X_train_final,
            "y_train_final": y_train_final,
            "X_kaggle": X_kaggle
        }
        