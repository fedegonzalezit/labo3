from typing import Dict, Optional, List, Tuple
from pipeline import PipelineStep, Pipeline
import pandas as pd
import numpy as np
from scipy.stats import linregress



class CreateSerieIdStep(PipelineStep):

    def execute(self, df: pd.DataFrame) -> Dict:
        """
        Creo una nueva columna serieID que es la combinacion entre serie_id y customer_id
        """
        df['serieID'] = df['product_id'].astype(str) + df['customer_id'].astype(str)
        df['serieID'] = df['serieID'].astype('uint64')
        # le resto el valor menimo asi empieza en 1
        df["serieID"] = df["serieID"] - df["serieID"].min() + 1
        # la paso a uint32
        df["serieID"] = df["serieID"].astype("uint32")
        return {"df": df}
    


class DropMinSerieMonthStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, months: int = 3):
        super().__init__(name)
        self.months = months

    def execute(self, df: pd.DataFrame) -> Dict:
        """
        Agrupo las series por customer_id y product_id, cuento el largo de cada serie y elimino las series que tienen menos de self.months meses
        """
        # Agrupo por serieID y cuento los meses únicos
        series_counts = df.groupby('serieID')['mes'].nunique()
        
        # Filtrar series con menos de self.months meses
        valid_series = series_counts[series_counts >= self.months].index
        
        # Filtrar el DataFrame original
        df = df[df['serieID'].isin(valid_series)]
        print(f"Number of series dropped : {len(series_counts) - len(valid_series)}")
        
        return {"df": df}
    

class FilterProductsIDStep(PipelineStep):
    def __init__(self, product_file = "product_id_apredecir201912.txt", dfs=["df"], name: Optional[str] = None):
        super().__init__(name)
        self.file = product_file
        self.dfs = dfs

    def execute(self, pipeline: Pipeline) -> None:
        """ el txt es un csv que tiene columna product_id separado por tabulaciones """
        converted_dfs = {}
        for df_key in self.dfs:
            df = pipeline.get_artifact(df_key)
            product_ids = pd.read_csv(self.file, sep="\t")["product_id"].tolist()
            df = df[df["product_id"].isin(product_ids)]
            converted_dfs[df_key] = df
            print(f"Filtered DataFrame {df_key} shape: {df.shape}")
        return converted_dfs
    

class FilterProductForTestingStep(PipelineStep):
    def __init__(self, total_products_ids: int = 100, name: Optional[str] = None, random=True):
        super().__init__(name)
        self.total_products_ids = total_products_ids
        self.random = random
        
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Filtra el DataFrame para que contenga solo los primeros total_products_ids productos """
        unique_products = df['product_id'].unique()
        if len(unique_products) > self.total_products_ids:
            if self.random:
                products = np.random.choice(unique_products, size=self.total_products_ids, replace=False)
            else:
                products = unique_products[:self.total_products_ids]
            df = df[df['product_id'].isin(products)]
        print(f"Filtered DataFrame shape: {df.shape}")
        return {"df": df}
    

class CastDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, df: pd.DataFrame) -> None:
        for col, dtype in self.dtypes.items():
            df[col] = df[col].astype(dtype)
        print(df.info())
        return {"df": df}
    

class ReduceMemoryUsageStep(PipelineStep):

    def execute(self, df):
        initial_mem_usage = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                if pd.api.types.is_float_dtype(df[col]):
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                elif pd.api.types.is_integer_dtype(df[col]):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
        
        final_mem_usage = df.memory_usage().sum() / 1024**2
        print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
        print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
        print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))
        return {"df": df}      

        
class ChangeDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, df) -> None:
        for original_dtype, dtype in self.dtypes.items():
            for col in df.select_dtypes(include=[original_dtype]).columns:
                df[col] = df[col].astype(dtype)
        print(df.info())
        return {"df": df}
    

class FilterFirstDateStep(PipelineStep):
    def __init__(self, first_date: str, name: Optional[str] = None):
        super().__init__(name)
        self.first_date = first_date

    def execute(self, df) -> None:
        df = df[df["fecha"] >= self.first_date]
        print(f"Filtered DataFrame shape: {df.shape}")
        return {"df": df}
    

class FeatureEngineeringLagStep(PipelineStep):
    def __init__(self, lags: List[int], columns: List, name: Optional[str] = None, column_rename= None):
        super().__init__(name)
        self.lags = lags
        self.columns = columns
        self.all = all
        self.column_rename = column_rename


    def execute(self, df: pd.DataFrame) -> dict:
        # Ordenar por grupo y fecha para que los lags sean correctos

        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        
        # Crear lags usando groupby y shift (vectorizado)
        grouped = df.groupby(['product_id', 'customer_id'])
        for column in self.columns:
            for lag in self.lags:
                c = self.column_rename or column
                df[f"{c}_lag_{lag}"] = grouped[column].shift(lag)
        return {"df": df}
    

import multiprocessing


class RollingMeanFeatureStep(PipelineStep):
    def __init__(self, windows: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_mean(self, args):
        col, window, df_small = args
        grouped = df_small.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).mean())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                # Solo pasar las columnas necesarias a cada proceso
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_mean, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}


class RollingStdFeatureStep(PipelineStep):
    def __init__(self, windows: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_std(self, args):
        col, window, df_small = args
        grouped = df_small.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_std_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).std())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                # Solo pasar las columnas necesarias a cada proceso
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_std, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}
    

class RollingSkewFeatureStep(PipelineStep):
    def __init__(self, windows: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_skew(self, args):
        col, window, df = args
        grouped = df.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_skew_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).skew())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_skew, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}
    

    

class RollingZscoreFeatureStep(PipelineStep):
    def __init__(self, windows: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_zscore(self, args):
        col, window, df = args
        grouped = df.groupby(['product_id', 'customer_id'])
        rolling_mean = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).mean())
        rolling_std = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).std())
        zscore = (df[col] - rolling_mean) / (rolling_std + 1e-6)
        return (f'{col}_rolling_zscore_{window}', zscore)

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                # Solo pasar las columnas necesarias a cada proceso
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_zscore, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}
    

class RollingAutocorrelationFeatureStep(PipelineStep):
    def __init__(self, window: int, columns: List[str], lags: List[int], name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns
        self.lags = lags

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        grouped = df.groupby(['product_id', 'customer_id'])
        
        for col in self.columns:
            for lag in self.lags:
                df[f'{col}_rolling_autocorr_{lag}_{self.window}'] = grouped[col].transform(
                    lambda x: x.rolling(self.window, min_periods=self.window,).apply(
                        lambda y: y.autocorr(lag=lag), raw=False
                    )
                )
        return {"df": df}


import multiprocessing

class RollingMaxFeatureStep(PipelineStep):
    def __init__(self, windows: int, columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_max(self, args):
        col, window, df = args
        grouped = df.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_max_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).max())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                # Solo pasar las columnas necesarias a cada proceso
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_max, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}
    

class RollingMinFeatureStep(PipelineStep):
    def __init__(self, windows: int, columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_min(self, args):
        col, window, df = args
        grouped = df.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_min_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).min())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_min, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}


class RollingStdFeatureStep(PipelineStep):
    def __init__(self, windows: int, columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_std(self, args):
        col, window, df = args
        grouped = df.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_std_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).std())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_std, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}
    

class ExponentialMovingAverageStep(PipelineStep):
    def __init__(self, span: int, columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.span = span
        self.columns = columns

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        grouped = df.groupby(['product_id', 'customer_id'])
        for col in self.columns:
            df[f'{col}_ema_{self.span}'] = grouped[col].transform(
                lambda x: x.ewm(span=self.span, adjust=False).mean()
            )
        return {"df": df}
    

class TrendFeatureStep(PipelineStep):
    def __init__(self, window: int, columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        
        def calculate_trend(series):
            return series.rolling(self.window).apply(
                lambda x: linregress(np.arange(len(x)), x)[0], raw=False
            )
        
        grouped = df.groupby(['product_id', 'customer_id'])
        for col in self.columns:
            df[f'{col}_trend_{self.window}'] = grouped[col].transform(calculate_trend)
        return {"df": df}
    

class DiffFeatureStep(PipelineStep):
    def __init__(self, periods: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.periods = periods
        self.columns = columns

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        grouped = df.groupby(['product_id', 'customer_id'])
        for col in self.columns:
            for period in self.periods:
                df[f'{col}_diff_{period}'] = grouped[col].diff(period)
        return {"df": df}
    
class DiffLogRatioFeatureStep(PipelineStep):
    def __init__(self, period, window: int, column:str, min_periods=None, name: Optional[str] = None):
        super().__init__(name)
        self.column = column
        self.window = window
        self.min_periods = min_periods if min_periods is not None else window
        self.period = period

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        base_prediction = (
            df.groupby(['product_id', 'customer_id'])[self.column]
            .transform(lambda x: x.rolling(self.window, min_periods=self.min_periods).mean())
        ).shift(self.period)
        # CORRECCIÓN: obtener la serie alineada
        column_to_diff = df.groupby(['product_id', 'customer_id'])[self.column].transform(lambda x: x)
        df[f"{self.column}_diff_log_ratio_{self.period}_{self.window}"] = (
            np.log((column_to_diff + 0.5) / (base_prediction + 0.5))
        )
        return {"df": df}
    

class RollingMedianFeatureStep(PipelineStep):
    def __init__(self, windows: int, columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows
        self.columns = columns

    def _compute_rolling_median(self, args):
        col, window, df = args
        grouped = df.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_median_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).median())
        )

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        tasks = []
        for col in self.columns:
            for window in self.windows:
                # Solo pasar las columnas necesarias a cada proceso
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._compute_rolling_median, tasks)

        for col_name, series in results:
            df[col_name] = series

        return {"df": df}
    

class CreateTotalCategoryStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, cat: str = "cat1", tn: str = "tn"):
        super().__init__(name)
        self.cat = cat
        self.tn = tn
    
    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(['fecha', self.cat])
        df[f"{self.tn}_{self.cat}_vendidas"] = (
            df.groupby(['fecha', self.cat])[self.tn]
              .transform('sum')
        )
        return {"df": df}


class CreateWeightByCustomerStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, df: pd.DataFrame) -> Dict:
        # Aseguramos orden estable (opcional, mejora legibilidad)
        df = df.sort_values(['fecha', 'customer_id'])
        
        # 1) Sumatoria de 'tn' por (fecha, customer_id) directamente en cada fila
        df['tn_customer_vendidas'] = (
            df.groupby(['fecha', 'customer_id'])['tn']
              .transform('sum')
        )
        # 2) Sumatoria total de 'tn' por fecha
        df['tn_total_vendidas'] = (
            df.groupby('fecha')['tn']
              .transform('sum')
        )
        # 3) Ratio
        df['customer_weight'] = df['tn_customer_vendidas'] / df['tn_total_vendidas']
        return {"df": df}
    

class CreateWeightByProductStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, df: pd.DataFrame) -> Dict:
        # Aseguramos orden estable (opcional, mejora legibilidad)
        df = df.sort_values(['fecha', 'product_id'])
        # 1) Sumatoria de 'tn' por (fecha, product_id) directamente en cada fila
        df['tn_product_vendidas'] = (
            df.groupby(['fecha', 'product_id'])['tn']
              .transform('sum')
        )
        # 2) Sumatoria total de 'tn' por fecha
        df['tn_total_vendidas'] = (
            df.groupby('fecha')['tn']
              .transform('sum')
        )
        # 3) Ratio
        df['product_weight'] = df['tn_product_vendidas'] / df['tn_total_vendidas']
        return {"df": df}
    

class CopyColumnAsWeightStep(PipelineStep):
    def __init__(self, column: str, name: Optional[str] = None):
        super().__init__(name)
        self.column = column

    def execute(self, df: pd.DataFrame) -> Dict:
        df[f"weight"] = np.abs(df[self.column])
        return {"df": df}
    

class WeightedProductIdStep(PipelineStep):
    """
    recibe un diccionario product_id: weight y agrega una columna weight donde el peso corresponde al peso del product_id de esa fila
    """
    def __init__(self, product_weights: Dict[str, float], name: Optional[str] = None):
        super().__init__(name)
        self.product_weights = product_weights

    def execute(self, df: pd.DataFrame) -> Dict:
        df['weight'] = df['product_id'].map(self.product_weights).fillna(0.0)
        return {"df": df}

class TimeDecayWeghtedProductIdStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, decay_factor: float = 0.99):
        super().__init__(name)
        self.decay_factor = decay_factor

    def execute(self, df: pd.DataFrame, train_index) -> Dict:
        # list of uniques date_id
        unique_train_dates = df.loc[train_index, 'date_id'].unique()
        # Create a decay factor based on the date_id
        decay_factor = self.decay_factor
        # Create a mapping of date_id to weight
        #date_weights = {date_id: decay_factor ** (len(unique_dates) - idx - 1) for idx, date_id in enumerate(unique_dates)}
        date_weights = {date_id: decay_factor ** (len(unique_train_dates) - idx - 1) for idx, date_id in enumerate(unique_train_dates)}
        # Map the weights to the DataFrame
        df['weight'] = df['date_id'].map(date_weights).fillna(1)
        return {"df": df}

class ManualDateIdWeightStep(PipelineStep):
    def __init__(self, date_weights: Dict[int, float], name: Optional[str] = None):
        super().__init__(name)
        self.date_weights = date_weights

    def execute(self, df: pd.DataFrame) -> Dict:
        # multiplica a df["weight"] por el peso de la fecha, si la columna weight no existe la crea
        # nota , no todas las fechas estan en date_weight, podria tener solo {31: 0.8, 32: 0.9} y las otras fechas mantienen su peso original sin multiplicar
        if "weight" not in df.columns:
            df["weight"] = 1.0
        df['weight'] *= df['date_id'].map(self.date_weights).fillna(1.0)
        return {"df": df}
        
class FeatureDivInteractionStep(PipelineStep):
    def __init__(self, columns: List[Tuple[str, str]], name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns

    def execute(self, df) -> None:
        for col1, col2 in self.columns:
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)  # Evitar división por cero
        return {"df": df}


class FeatureProdInteractionStep(PipelineStep):
    def __init__(self, columns: List[Tuple[str, str]], name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns

    def execute(self, df) -> None:
        for col1, col2 in self.columns:
            df[f"{col1}_prod_{col2}"] = df[col1] * df[col2]
        return {"df": df}
    

class DateRelatedFeaturesStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, df) -> None:
        df["year"] = df["fecha"].dt.year
        df["mes"] = df["fecha"].dt.month
        df["quarter"] = df["fecha"].dt.quarter
        unique_dates = df['fecha'].drop_duplicates().sort_values()
        date_to_id = {date: idx for idx, date in enumerate(unique_dates)}
        df['date_id'] = df['fecha'].map(date_to_id).astype('uint8')
        df['date_id'] = df['date_id'].astype('uint8')
        df["dias_del_mes"] = df["fecha"].dt.daysinmonth
        # creo variables coseno y seno para la fecha
        df["coseno_fecha"] = np.cos(2 * np.pi * df["date_id"] / len(unique_dates))
        df["seno_fecha"] = np.sin(2 * np.pi * df["date_id"] / len(unique_dates))
        


        return {"df": df}
