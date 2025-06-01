import pandas as pd
from abc import ABC, abstractmethod
from pipeline import PipelineStep
from typing import Dict, Optional


class PipelineScaler(ABC):
    def __init__(self, column: str):
        self.column = column
        self.scaler_data = None

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

class PipelineStandarScaler(PipelineScaler):

    def fit(self, df: pd.DataFrame):
        agg = df.groupby(['product_id', 'customer_id'])[self.column].agg(['mean', 'std']).rename(
            columns={'mean': f'{self.column}_mean_scaler', 'std': f'{self.column}_std_scaler'})
        self.scaler_data = agg
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # agrego columnas temporales
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df[f'{self.column}_scaled'] = (df[self.column] - df[f'{self.column}_mean_scaler']) / (df[f'{self.column}_std_scaler'])
        # replace inf and -inf with NaN
        df[f'{self.column}_scaled'].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df[f'{self.column}_scaled'] = df[f'{self.column}_scaled'].fillna(0)
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_mean_scaler', f'{self.column}_std_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}_scaled"]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # agrego columnas temporales
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df[f"{self.column}_scaled"] = (df[f'{self.column}_scaled'] * (df[f'{self.column}_std_scaler'])) + df[f'{self.column}_mean_scaler']
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_mean_scaler', f'{self.column}_std_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}_scaled"]
    

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
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df[f'{self.column}_scaled'] = (df[self.column] - df[f'{self.column}_min_scaler']) / (df[f'{self.column}_max_scaler'] - df[f'{self.column}_min_scaler'])
        df[f'{self.column}_scaled'] = df[f'{self.column}_scaled'].fillna(0)
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_min_scaler', f'{self.column}_max_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}_scaled"]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scaler_data is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # agrego columnas temporales
        df = df.merge(self.scaler_data, on=['product_id', 'customer_id'], how='left')
        df[f"{self.column}_scaled"] = (df[f'{self.column}_scaled'] * (df[f'{self.column}_max_scaler'] - df[f'{self.column}_min_scaler'])) + df[f'{self.column}_min_scaler']
        # elimino las columnas temporales
        df.drop(columns=[f'{self.column}_min_scaler', f'{self.column}_max_scaler'], inplace=True, errors='ignore')
        return df[f"{self.column}_scaled"]
    

class ScaleFeatureStep(PipelineStep):
    def __init__(self, column: str, scaler=PipelineStandarScaler, name = None):
        super().__init__(name)
        self.column = column
        self.scaler_cls = scaler

    def execute(self, df: pd.DataFrame, train_index) -> Dict:
        scaler = self.scaler_cls(
            column=self.column,
        )
        train_df = df.loc[train_index]
        scaler.fit(train_df)
        df[f"{self.column}_scaled"] = scaler.transform(df)
        ret = {"df": df, f"scaler_{self.column}_scaled": scaler}
        return ret
    

class InverseScalePredictionsStep(PipelineStep):
    def execute(self, pipeline, target_col, predictions, df, y_test, test_index) -> Dict:
        """
        Inverse scale the predictions using the provided grouped scaler.
        """
        try:
            scaler = pipeline.get_artifact(f"scaler_{target_col}")
        except:
            scaler = None
        if scaler is None:
            print(f"Scaler for {target_col} not found in pipeline.")
            return
        if not isinstance(predictions, pd.Series):
            raise ValueError("Predictions must be a pandas Series.")
        
        # creo un df predictions_df que tiene predictions, product_id y customer_id de df para los indices de predictions
        predictions_df = pd.DataFrame(predictions, index=predictions.index)
        predictions_df.columns = [target_col]
        predictions_df["product_id"] = df["product_id"]
        predictions_df["customer_id"] = df["customer_id"]
        predictions = scaler.inverse_transform(predictions_df)
        
        test_df = df.loc[test_index][["product_id", "customer_id"]]
        test_df[target_col] = y_test
        test_df["target"] = scaler.inverse_transform(test_df[[target_col, 'product_id', 'customer_id']])    
 
        return {
            "predictions": predictions,
            "y_test": test_df["target"]
        }