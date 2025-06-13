from pipeline import Pipeline
from pipeline.steps import *
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
# suprimir logs de performance warnings de pandas
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.internals.managers")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

from pandas.errors import PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)

BASE_PATH = "./"
xgb_params = {
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 512,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbose": 0,
}

class FeatureEngineeringProductCatInteractionStep(PipelineStep):

    def __init__(self, cat="cat1", name: Optional[str] = None, tn="tn"):
        super().__init__(name)
        self.cat = cat
        self.tn = tn


    def execute(self, df, df_original=None) -> None:
        # agrupo el dataframe por cat1 (sumando), obteniendo fecha, cat1 y
        # luego paso el dataframe a wide format, donde cada columna es una categoria  y la fila es la suma de tn para cada cat1
        # luego mergeo al dataframe original por fecha y product_id
        df_index = df.index
        if df_original is None:
            df_to_proces = df
        else:
            df_to_proces = df_original
        df_cat = df_to_proces.groupby(["date_id", self.cat]).agg({self.tn: "sum"}).reset_index()
        print(df_cat)
        # cast column self.cat to string
        df_cat[self.cat] = df_cat[self.cat].astype(str)
        df_cat = df_cat.pivot(index="date_id", columns=self.cat, values=self.tn).reset_index()
        # paso a string los nombres de las columnas
        df_cat.columns = [f"{self.tn}_{self.cat}_{col}" if col != "date_id" else "date_id" for col in df_cat.columns]
        df = df.merge(df_cat, on="date_id", how="left")
        # vuelvo a setear el indice original
        df.index = df_index
        return {"df": df}
    

import numpy as np
class Log1pTranformation(PipelineStep):
    def execute(self, df):
        df["tn"] = df["tn"].apply(lambda x: np.log1p(x) if x >= 0 else 0)
        return {"df": df}
    
class InverseLog1pTranformation(PipelineStep):
    def execute(self, df, predictions, y_test):
        df["target"] = df["target"].apply(lambda x: np.expm1(x) if x >= 0 else 0)
        predictions = predictions.apply(lambda x: np.expm1(x) if x >= 0 else 0)
        y_test["target"] = y_test["target"].apply(lambda x: np.expm1(x) if x >= 0 else 0)
        return {"df": df, "predictions": predictions, "y_test": y_test}

class GroupByProductStep(PipelineStep):
    def execute(self, df) -> None:
        # Agrupo el dataframe por product_id y fecha, sumando las cantidades
        df = df.groupby(["product_id", "fecha"]).agg({
            'cust_request_qty': 'sum',
            'cust_request_tn': 'sum',
            'tn': 'sum',
            'stock_final': 'max',
            'cat1': 'first',
            'cat2': 'first',
            'cat3': 'first',
            'brand': 'first',
            'sku_size': 'max',
        }).reset_index()
        # le dejo la columna customer_id = 0 para que no crashee el resto
        df['customer_id'] = 1
        return {"df": df}

class FilterDatasetByColumn(PipelineStep):
    def __init__(self, column: str, value, name: Optional[str] = None):
        super().__init__(name)
        self.column = column
        self.value = value
        
    def execute(self, df) -> None:
        # Filtra el DataFrame por el valor de la columna especificada
        df_original = df.copy()
        df_filtered = df[df[self.column] == self.value]
        print(df_filtered.shape)
        return {"df": df_filtered, "df_original": df_original}

class DeleteBadColumns(PipelineStep):
    def execute(self, df) -> None:
        # Elimina  las columnas donde toda sus filas son NaN
        base_columns = df.columns
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, (df != 0).any(axis=0)]
        deleted_columns = set(base_columns) - set(df.columns)
        print(f"Deleted columns: {deleted_columns}")
        return {"df": df}

class CreateResidualTargetStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, target_col: str = 'tn', window: int = 12, min_periods = None):
        super().__init__(name)
        self.target_col = target_col
        self.window = window
        self.min_periods = min_periods or window

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(['product_id', 'customer_id', 'fecha']).copy()

        # Valor futuro (shift -2) que será la predicción final
        df['target_shifted'] = df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-2)

        # Cálculo del valor base usando la media rolling pasada
        df['base_prediction'] = (
            df.groupby(['product_id', 'customer_id'])[self.target_col]
            .transform(lambda x: x.rolling(self.window, min_periods=self.min_periods).mean())
        )

        # Target: residuo a predecir
        df['target'] = np.log((df['target_shifted']+0.5) / (df['base_prediction']+0.5))
        df.drop(columns=['target_shifted', 'base_prediction'], inplace=True, errors='ignore')

        return {
            "df": df,
            "target_col": self.target_col,
        }


class InverseResidualTargetStep(PipelineStep):
    def __init__(
        self,
        name: Optional[str] = None,
        target_col: str = 'tn',
        window: int = 12
    ):
        super().__init__(name)
        self.target_col = target_col
        self.window = window

    def execute(self, df: pd.DataFrame, predictions, y_test) -> Dict:

        # Orden correcto
        df = df.sort_values(['product_id', 'customer_id', 'fecha'])

        # Recalcular base_prediction exactamente como en CreateResidualTargetStep
        df['base_prediction'] = (
            df.groupby(['product_id', 'customer_id'])[self.target_col]
            .transform(lambda x: x.rolling(self.window, min_periods=1).mean())
        )

        # Invertir el target residual: target_original = residual + base_prediction
        df['target'] = (np.exp(df['target']) * (df['base_prediction']+0.5))-0.5

        predictions = (np.exp(predictions) * (df['base_prediction'].loc[y_test.index]+0.5))-0.5

        y_test = df[["target"]].loc[y_test.index]
        # Limpiar si no querés guardar la base
        df.drop(columns=['base_prediction'], inplace=True)

        return {"df": df, "predictions": predictions, "y_test": y_test}
    
# extraer features con prophet de df_filtered["tn"]
from prophet import Prophet
import numpy as np
import pandas as pd
from tqdm import tqdm

PROPHET_FEATURES = [
    "trend", "yhat_lower", "yhat_upper", "trend_lower", "trend_upper",
    "additive_terms", "additive_terms_lower", "additive_terms_upper",
    "yearly", "yearly_lower", "yearly_upper",
    "multiplicative_terms", "multiplicative_terms_lower", "multiplicative_terms_upper",
    "yhat"
]

def extract_prophet_features(df):
    df_prophet = df[["fecha", "tn"]].rename(columns={"fecha": "ds", "tn": "y"})
    df_prophet["ds"] = df_prophet["ds"].astype("datetime64[ns]")
    try:
        model = Prophet()
        model.fit(df_prophet)
        forecast = model.predict(df_prophet)
        features = pd.DataFrame(index=forecast.index)
        for col in PROPHET_FEATURES:
            features[col] = forecast[col] if col in forecast.columns else np.nan
    except Exception:
        features = pd.DataFrame({
            **{col: np.nan for col in PROPHET_FEATURES}
        })
    return features

class ProphetFeatureExtractionStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, df) -> pd.DataFrame:
        # Procesar por product_id y customer_id
        group_cols = ["product_id", "customer_id"]
        prophet_features = []
        for (pid, cid), group in tqdm(df.groupby(group_cols), desc="Prophet features"):
            # Solo si hay suficientes datos
            if group["tn"].notna().sum() > 2:
                features = extract_prophet_features(group)
            else:
                # Si no hay suficientes datos, devolver NaN
                features = pd.DataFrame({
                    "date_id": group["date_id"].values,
                    "product_id": pid,
                    "customer_id": cid,
                    **{col: np.nan for col in PROPHET_FEATURES}
                })
            prophet_features.append(features)
        prophet_features_df = pd.concat(prophet_features, ignore_index=True)
        # Merge eficiente por claves
        df = df.merge(
            prophet_features_df,
            on=["date_id", "product_id", "customer_id"],
            how="left"
        )
        return {"df": df}

import optuna

class OptunaObjectiveStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, pipeline_to_optimize: Pipeline = None, feature_selection: bool = False, n_trials=50):
        self.pipeline_to_optimize = pipeline_to_optimize
        self.feature_selection = feature_selection
        self.n_trials = n_trials
        super().__init__(name)

    def execute(self, df, train_index):
        def objective(trial):
            train_df = df.loc[train_index]
            # lgb params trial - regularización fuerte y subsampling para evitar overfitting
            lgb_params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "n_jobs": -1,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_child_samples": trial.suggest_int("min_child_samples", 2, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 1e-3, 5.0, log=True),
                "max_bin": trial.suggest_int("max_bin", 128, 700),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 100),
                "n_estimators": trial.suggest_int("n_estimators", 100, 3500),
            }
            # Feature selection opcional
            if self.feature_selection:
                features = [col for col in train_df.columns if col not in ["fecha", "target", "weight", "product_id", "customer_id", "date_id", "tn"]]
                feature_selection = {feature: trial.suggest_categorical(feature, [0, 1]) for feature in features}
                selected_features = [feature for feature, selected in feature_selection.items() if selected == 1]
                train_df = train_df[selected_features + ["fecha", "target", "weight",  "product_id", "customer_id", "date_id", "tn"]]
            else:
                feature_selection = None

            # Actualiza los parámetros del pipeline
            self.pipeline_to_optimize.save_artifact("df", train_df)
            self.pipeline_to_optimize.save_artifact("params", lgb_params)
            # Ejecuta el pipeline
            self.pipeline_to_optimize.run()
            total_error = self.pipeline_to_optimize.get_artifact("total_error")
            self.pipeline_to_optimize.clear()
            return total_error

        study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", load_if_exists=False, study_name="labo3_2")
        study.optimize(objective, n_trials=self.n_trials)
        best_lgb_params = study.best_params

        return {"params": study.best_params}


# features de analisis tecnico de series financieras pero que pueden funcionar
import ta

import multiprocessing

class TechnicalAnalysisFeaturesStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, column: str = 'tn'):
        super().__init__(name)
        self.column = column

    def _run_ta_wrapper(self, group):
        return self.run_ta(group.copy())

    def execute(self, df) -> pd.DataFrame:
        grouped = [group for _, group in df.groupby(["product_id", "customer_id"])[["date_id", "product_id", "customer_id", self.column]]]
        # Procesar en paralelo usando todos los cores disponibles
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            dfs = pool.map(self._run_ta_wrapper, grouped)
        df_ta = pd.concat(dfs, axis=0)
        # Solo las columnas nuevas
        new_cols = [col for col in df_ta.columns if col not in df.columns]
        # Merge por claves
        df = df.merge(
            df_ta[["date_id", "product_id", "customer_id"] + new_cols],
            on=["date_id", "product_id", "customer_id"],
            how="left"
        )
        return {"df": df}
    
    
    def run_ta(self, df) -> pd.DataFrame:
        # Asegurarse de que 'fecha' sea un índice de tipo datetime
        # momentum
        df[f"{self.column}_kama_indicator"] = ta.momentum.KAMAIndicator(df[self.column], window=10).kama()
        #PercentagePriceOscillator
        df[f"{self.column}_ppo"] = ta.momentum.PercentagePriceOscillator(df[self.column], window_slow=12, window_fast=3).ppo()
        df[f"{self.column}_ppo_signal"] = ta.momentum.PercentagePriceOscillator(df[self.column], window_slow=12, window_fast=3).ppo_signal()
        df[f"{self.column}_ppo_hist"] = ta.momentum.PercentagePriceOscillator(df[self.column], window_slow=12, window_fast=3).ppo_hist()
        #ROCIndicator
        df[f"{self.column}_roc"] = ta.momentum.ROCIndicator(df[self.column], window=12).roc()
        #RSIIndicator
        df[f"{self.column}_rsi"] = ta.momentum.RSIIndicator(df[self.column], window=6).rsi()
        # StochRSIIndicator
        df[f"{self.column}_stoch_rsi"] = ta.momentum.StochRSIIndicator(df[self.column], window=7, smooth1=3, smooth2=3).stochrsi()
        df[f"{self.column}_stoch_rsi_k"] = ta.momentum.StochRSIIndicator(df[self.column], window=7, smooth1=3, smooth2=3).stochrsi_k()
        df[f"{self.column}_stoch_rsi_d"] = ta.momentum.StochRSIIndicator(df[self.column], window=7, smooth1=3, smooth2=3).stochrsi_d()
        #TSIIndicator
        df[f"{self.column}_tsi"] = ta.momentum.TSIIndicator(df[self.column], window_slow=10, window_fast=4).tsi()
        #BollingerBands
        df[f"{self.column}_bollinger_hband"] = ta.volatility.BollingerBands(df[self.column], window=12, window_dev=2).bollinger_hband()
        df[f"{self.column}_bollinger_lband"] = ta.volatility.BollingerBands(df[self.column], window=12, window_dev=2).bollinger_lband()
        df[f"{self.column}_bollinger_mavg"] = ta.volatility.BollingerBands(df[self.column], window=12, window_dev=2).bollinger_mavg()
        df[f"{self.column}_bollinger_wband"] = ta.volatility.BollingerBands(df[self.column], window=12, window_dev=2).bollinger_wband()
        df[f"{self.column}_bollinger_pband"] = ta.volatility.BollingerBands(df[self.column], window=12, window_dev=2).bollinger_pband()
        # UlcerIndex
        df[f"{self.column}_ulcer_index"] = ta.volatility.UlcerIndex(df[self.column], window=8).ulcer_index()
        # DPOIndicator
        df[f"{self.column}_dpo"] = ta.trend.DPOIndicator(df[self.column], window=8).dpo()
        #MACD
        df[f"{self.column}_macd"] = ta.trend.MACD(df[self.column], window_slow=13, window_fast=6, window_sign=3).macd()
        df[f"{self.column}_macd_signal"] = ta.trend.MACD(df[self.column], window_slow=13, window_fast=6, window_sign=3).macd_signal()
        df[f"{self.column}_macd_diff"] = ta.trend.MACD(df[self.column], window_slow=13, window_fast=6, window_sign=3).macd_diff()
        # TRIXIndicator
        df[f"{self.column}_trix"] = ta.trend.TRIXIndicator(df[self.column], window=12).trix()
        return df
    
class OperationBetweenColumnsStep(PipelineStep):
    def __init__(self, column1: str, column2: str, operation: str, new_column_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.column1 = column1
        self.column2 = column2
        self.operation = operation
        self.new_column_name = new_column_name

    def execute(self, df) -> pd.DataFrame:
        if self.operation == "add":
            df[self.new_column_name] = df[self.column1] + df[self.column2]
        elif self.operation == "subtract":
            df[self.new_column_name] = df[self.column1] - df[self.column2]
        elif self.operation == "multiply":
            df[self.new_column_name] = df[self.column1] * df[self.column2]
        elif self.operation == "divide":
            df[self.new_column_name] = df[self.column1] / (df[self.column2] + 1e-9)  # Avoid division by zero
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
        
        return {"df": df}
    
params = {'learning_rate': 0.09249276920204523,
 'num_leaves': 17,
 'max_depth': 18,
 'min_child_samples': 139,
 'subsample': 0.8665108176379659,
 'colsample_bytree': 0.7445887789494118,
 'reg_alpha': 0.0909029253091313,
 'reg_lambda': 0.22154388770108127,
 'min_split_gain': 0.0033066931135108304,
 'max_bin': 348,
 'feature_fraction': 0.14374611747734473,
 'bagging_fraction': 0.8972458727623152,
 'extra_trees': False,
 'bagging_freq': 45,
 'n_estimators': 2718}
 #'n_estimators': 100}


TEST_DATE=35
EXPERIMENT="EXP_LOCO"

models_list = []
model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(target_col="tn_rolling_12", adj_value=1000), # MUY sensible al valor de este epsilon
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(target_col="tn_rolling_12", adj_value=1000),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)


model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(target_col="tn_rolling_12", adj_value=100), # MUY sensible al valor de este epsilon
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(target_col="tn_rolling_12", adj_value=100),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)

model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(target_col="tn_rolling_3", adj_value=50),
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(target_col="tn_rolling_3", adj_value=50),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)


# hago un ensamble de modelos 3 modelos con distintos date_id_start
model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(target_col="tn_rolling_3", adj_value=10),
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(target_col="tn_rolling_3", adj_value=10),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)

# hago un ensamble de modelos 3 modelos con distintos date_id_start
model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(target_col="tn_rolling_3", adj_value=5),
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(target_col="tn_rolling_3", adj_value=5),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)



model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn_scaled"),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)

model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn_scaled"),
        TransformTargetDiffStep(),
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformDiffStep(),
        InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)

# hago un ensamble de modelos 3 modelos con distintos date_id_start
model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(),
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline)

# hago un ensamble de modelos 3 modelos con distintos date_id_start
model_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep("df_fe_epic_light.pickle"),
        SplitDataFrameStep2(df="df", test_date=33, gap=1),
        TimeDecayWeghtedProductIdStep(decay_factor=0.99),
        # marco outliers
        #ManualDateIdWeightStep(date_weights={
        #    29: 0.5,
        #    30: 0.7,
        #    31: 0.8
        #}),
        TrainScalerFeatureStep(column="tn"),
        TrainScalerFeatureStep(column="cust_request_qty"),
        TransformScalerFeatureStep(column=r'tn(?!.*(_div_|_per_|_minus_|_prod_))', regex=True, scaler_name="scaler_tn"),
        TransformScalerFeatureStep(column="cust_request_qty", scaler_name="scaler_cust_request_qty"),
        CreateTargetColumStep(target_col="tn"),
        TransformTargetLog1pDiffStep(target_col="tn_rolling_3"),
        # creo una columna lag_2 del target que es la serie historica
        FeatureEngineeringLagStep(lags=[2], columns=["target"]),
        # vuelvo a hacer FE de la nueva serie historica :)
        FeatureEngineeringLagStep(lags=list(range(1,25)), columns=["target_lag_2"]),
        RollingMeanFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingStdFeatureStep(windows=list(range(3,25)), columns=["target_lag_2"]),
        RollingMaxFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        RollingMinFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingSkewFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),
        #RollingZscoreFeatureStep(windows=list(range(2,25)), columns=["target_lag_2"]),

        DiffFeatureStep(periods=list(range(1,25)), columns=["target_lag_2"]),
        PrepareXYStep(),
        TrainModelStep(params=params),
        PredictStep(),
        InverseTransformLog1pDiffStep(target_col="tn_rolling_3"),
        #InverseTransformScalerFeatureStep(column="target", scaler_name="scaler_tn"),
        #InverseTransformScalerFeatureStep(column="predictions", scaler_name="scaler_tn"),
        EvaluatePredictionsSteps(filter_file="product_id_apredecir201912.txt"),
        PlotFeatureImportanceStep(),
        
    
    ],
    optimize_arftifacts_memory=False,

)
model_pipeline.run() 
models_list.append(model_pipeline) 

predictions = []
for m in models_list:
    pred = m.get_artifact("eval_df")["tn_pred"]
    print(pred)
    predictions.append(pred)
prom_predictions = sum(predictions) / len(predictions)
prom_predictions

eval_df = models_list[3].get_artifact("eval_df").copy()
eval_df["prom_predictions"] = prom_predictions

# los prom_predictions que sean menor a 0 los seteo a 0
eval_df["prom_predictions"] = eval_df["prom_predictions"].apply(lambda x: max(x, 0))
eval_df["error_prom_abs"] = np.abs(eval_df["prom_predictions"] - eval_df["tn_real"])
eval_df.sort_values("error_prom_abs", ascending=False, inplace=True)

total_error = np.sum(eval_df["error_prom_abs"]) / np.sum(eval_df["tn_real"])
print(f"Total error: {total_error:.4f}")