import pandas as pd
from abc import ABC, abstractmethod
from pipeline import PipelineStep
from typing import Dict, Optional


class PipelineScaler(ABC):
    def __init__(self, column: str, group_by=['product_id', 'customer_id']):
        self.column = column
        self.scaler_data = None
        self.group_by = group_by 

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.Series:
        pass

# TODO: hacer transformacion log1p si es necesario
# TODO: debuggear, por alguna razon da mal

class PipelineRobustScaler(PipelineScaler):
    
    def fit(self, df: pd.DataFrame):
        grouped = df.groupby(['product_id', 'customer_id'])[self.column]  # SeriesGroupBy
        median = grouped.median()
        q1 = grouped.apply(lambda x: x.quantile(0.25))
        q3 = grouped.apply(lambda x: x.quantile(0.75))
        iqr = q3 - q1

        agg = pd.DataFrame({
            f'{self.column}_median_scaler': median,
            f'{self.column}_iqr_scaler': iqr
        })
        print(agg.head())
        self.scaler_data = agg
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        original_index = df.index
        original_nans = df[self.column].isna()
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df.set_index(original_index, inplace=True)
        df[f'{self.column}_scaled'] = (df[self.column] - df[f'{self.column}_median_scaler']) / (df[f'{self.column}_iqr_scaler'])
        # replace inf and -inf with NaN
        df[f'{self.column}_scaled'].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df[f'{self.column}_scaled'] = df[f'{self.column}_scaled'].fillna(0)
        # original nans
        df[f'{self.column}_scaled'] = df[f'{self.column}_scaled'].where(~original_nans, other=pd.NA)
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_median_scaler', f'{self.column}_iqr_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}_scaled"]

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")

        # agrego columnas temporales
        df_index = df.index
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        # reconstruyo los indices originales
        df.set_index(df_index, inplace=True)
        df[f"{self.column}"] = (df[f'{self.column}'] * (df[f'{self.column}_iqr_scaler'])) + df[f'{self.column}_median_scaler']
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_median_scaler', f'{self.column}_iqr_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}"]


class PipelineStandarScaler(PipelineScaler):

    def fit(self, df: pd.DataFrame):
        agg = df.groupby(self.group_by)[self.column].agg(['mean', 'std']).rename(
            columns={'mean': f'{self.column}_mean_scaler', 'std': f'{self.column}_std_scaler'})
        self.scaler_data = agg
        #self.scaler_data.fillna(0, inplace=True)
        return self
    
    def transform(self, df: pd.DataFrame, column="") -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        column = column or self.column
        # agrego columnas temporales
        original_index = df.index
        original_nans = df[column].isna()
        df = df.merge(self.scaler_data, on=self.group_by, how='left')
        df.set_index(original_index, inplace=True)
        df[f'{column}_scaled'] = (df[column] - df[f'{self.column}_mean_scaler']) / (df[f'{self.column}_std_scaler'])
        # replace inf and -inf with NaN
        df[f'{column}_scaled'].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        # original nans
        # hago un fill nan de las rows que no eran nan en la serie original
        df[f'{column}_scaled'] = df[f'{column}_scaled'].fillna(0)
        df[f'{column}_scaled'] = df[f'{column}_scaled'].where(~original_nans, other=pd.NA)
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_mean_scaler', f'{self.column}_std_scaler'], inplace=True, errors='ignore')
        return df[f"{column}_scaled"]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame, column="") -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")

        column = column or self.column
        # agrego columnas temporales
        df_index = df.index
        df = df.merge(self.scaler_data, on=self.group_by, how='left')
        # reconstruyo los indices originales
        df.set_index(df_index, inplace=True)
        df[f"{column}"] = (df[f'{column}'] * (df[f'{self.column}_std_scaler'])) + df[f'{self.column}_mean_scaler']
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_mean_scaler', f'{self.column}_std_scaler'], inplace=True, errors='ignore')
        return df[f"{column}"]
    

class PipelineMinMaxScaler(PipelineScaler):

    def fit(self, df: pd.DataFrame):
        agg = df.groupby(['product_id', 'customer_id'])[self.column].agg(['min', 'max']).rename(
            columns={'min': f'{self.column}_min_scaler', 'max': f'{self.column}_max_scaler'})
        # seteo el minimo con 0 asi queda estandarlizado en todas las series
        agg[f'{self.column}_min_scaler'] = 0
        self.scaler_data = agg
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # agrego columnas temporales
        original_index = df.index
        original_nans = df[self.column].isna()
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df.set_index(original_index, inplace=True)
        df[f'{self.column}_scaled'] = (df[self.column] - df[f'{self.column}_min_scaler']) / (df[f'{self.column}_max_scaler'] - df[f'{self.column}_min_scaler'])
        df[f'{self.column}_scaled'].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        # original nans
        df[f'{self.column}_scaled'] = df[f'{self.column}_scaled'].fillna(0)
        df[f'{self.column}_scaled'] = df[f'{self.column}_scaled'].where(~original_nans, other=pd.NA)
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_min_scaler', f'{self.column}_max_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}_scaled"]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # agrego columnas temporales
        df_index = df.index
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df.set_index(df_index, inplace=True)

        df[f"{self.column}"] = (df[f'{self.column}'] * (df[f'{self.column}_max_scaler'] - df[f'{self.column}_min_scaler'])) + df[f'{self.column}_min_scaler']
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_min_scaler', f'{self.column}_max_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}"]
    
import numpy as np
class Log1pStep(PipelineStep):
    def __init__(self, name=None, column="target"):
        super().__init__(name)
        self.column = column

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(["customer_id", "product_id"])
        df[f"{self.column}"] = grouped[self.column].transform(lambda x: x + 1).apply(np.log1p)
        return df
    

class ScaleFeatureStep(PipelineStep):
    def __init__(self, column: str, regex=False, override=False, scaler=PipelineStandarScaler, name = None):
        super().__init__(name)
        self.column = column
        self.scaler_cls = scaler
        self.regex = regex
        self.override = override

    def execute(self, df: pd.DataFrame, train_scaler_index) -> Dict:
        # si regex es True, busco todas las columnas que coincidan con el regex
        if self.regex:
            columns = df.filter(regex=self.column, axis=1).columns.tolist()
            print(f"Columns found matching regex '{self.column}': {columns}")
            if not columns:
                raise ValueError(f"No columns found matching regex '{self.column}'")
        else:
            columns = [self.column]
        scalers = {}
        for column in columns:
            scaler = self.scaler_cls(
                column=column,
            )
            if self.override:
                column_scaled = column
            else:
                column_scaled = f"{column}_scaled"
            train_df = df.loc[train_scaler_index]
            scaler.fit(train_df[["product_id", "customer_id", column]])
            df[column_scaled] = scaler.transform(df[["product_id", "customer_id", column]])
            scalers[f"scaler_{column_scaled}"] = scaler
        ret = {"df": df, **scalers}
        return ret


class TrainScalerFeatureStep(PipelineStep):
    def __init__(self, column: str, scaler=PipelineStandarScaler, name=None):
        super().__init__(name)
        self.column = column
        self.scaler_cls = scaler

    def execute(self, df: pd.DataFrame, train_scaler_index) -> Dict:
        # si regex es True, busco todas las columnas que coincidan con el regex
        scaler = self.scaler_cls(
            column=self.column,
            group_by=["product_id"]
        )
        train_df = df.loc[train_scaler_index]
        scaler.fit(train_df[["product_id", self.column]])
        return {
            f"scaler_{self.column}": scaler
        }


class TransformScalerFeatureStep(PipelineStep):
    def __init__(self, column: str, scaler_name: str, regex=False, name=None, override=False):
        super().__init__(name)
        self.column = column
        self.scaler_name = scaler_name
        self.override = override
        self.regex = regex

    def execute(self, pipeline, df: pd.DataFrame) -> Dict:
        if self.regex:
            columns = df.filter(regex=self.column, axis=1).columns.tolist()
            print(f"Columns found matching regex '{self.column}': {columns}")
            if not columns:
                raise ValueError(f"No columns found matching regex '{self.column}'")
        else:
            columns = [self.column]
        for column in columns:
            if self.override:
                column_scaled = column
            else:
                column_scaled = f"{column}_scaled"
        
            scaler = pipeline.get_artifact(self.scaler_name)        
            df[column_scaled] = scaler.transform(df[["product_id", column]], column=column)
        return {
            "df": df,
        }

class InverseTransformScalerFeatureStep(PipelineStep):
    def __init__(self, column: str, scaler_name: str, regex=False, name=None):
        super().__init__(name)
        self.column = column
        self.scaler_name = scaler_name
        self.regex = regex

    def execute(self, pipeline, df: pd.DataFrame) -> Dict:
        if self.regex:
            columns = df.filter(regex=self.column, axis=1).columns.tolist()
            print(f"Columns found matching regex '{self.column}': {columns}")
            if not columns:
                raise ValueError(f"No columns found matching regex '{self.column}'")
        else:
            columns = [self.column]
        for column in columns:
            scaler = pipeline.get_artifact(self.scaler_name)
            df[column] = scaler.inverse_transform(df[["product_id", column]], column=column)
        return {
            "df": df,
        }

class InverseScalePredictionsStep(PipelineStep):
    def execute(self, predictions, df, test_index, scaler_target=None) -> Dict:
        """
        Inverse scale the predictions using the provided grouped scaler.
        """
        if not scaler_target:
            return

        # creo un df predictions_df que tiene predictions, product_id y customer_id de df para los indices de predictions
        predictions_df = pd.DataFrame(predictions, index=predictions.index)
        predictions_df["product_id"] = df["product_id"]
        predictions_df["customer_id"] = df["customer_id"]
        predictions_df.columns = ["target", "product_id", "customer_id"]
        predictions = scaler_target.inverse_transform(predictions_df)
        predictions = pd.Series(predictions, name="predictions")
        predictions.index = test_index
        predictions.fillna(0, inplace=True)

        df["target"] = scaler_target.inverse_transform(df[["target", "product_id", "customer_id"]])    
 
        return {
            "predictions": predictions,
            "df": df
        }