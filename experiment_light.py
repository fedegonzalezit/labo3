from pipeline import Pipeline
from pipeline.steps import *

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
    def __init__(self, name: Optional[str] = None, target_col: str = 'tn', window: int = 12):
        super().__init__(name)
        self.target_col = target_col
        self.window = window

    def execute(self, df: pd.DataFrame) -> Dict:
        df = df.sort_values(['product_id', 'customer_id', 'fecha']).copy()

        # Valor futuro (shift -2) que será la predicción final
        df['target_shifted'] = df.groupby(['product_id', 'customer_id'])[self.target_col].shift(-2)

        # Cálculo del valor base usando la media rolling pasada
        df['base_prediction'] = (
            df.groupby(['product_id', 'customer_id'])[self.target_col]
            .transform(lambda x: x.rolling(self.window, min_periods=1).mean())
        )

        # Target: residuo a predecir
        df['target'] = df['target_shifted'] - df['base_prediction']
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
        df['target'] = df['target'] + df['base_prediction']

        predictions = predictions + df['base_prediction'].loc[y_test.index]

        y_test = df[["target"]].loc[y_test.index]
        # Limpiar si no querés guardar la base
        df.drop(columns=['base_prediction'], inplace=True)

        return {"df": df, "predictions": predictions, "y_test": y_test}



pipeline = Pipeline(
    steps=[
        LoadDataFrameStep("df_intermedio.parquet"),
        #GroupByProductStep(),
        FilterProductsIDStep(dfs=["df"]),
        #FilterDatasetByColumn(column="cat1", value="FOODS"),
        DateRelatedFeaturesStep(),
        #Log1pTranformation(),
        #CreateTargetColumDiffStep(target_col="tn"),
        #CreateTargetColumStep(target_col="tn"),
        CreateResidualTargetStep(target_col="tn", window=12),
        SplitDataFrameStep2(df="df", test_date=29, gap=1),
        ReduceMemoryUsageStep(),

        #ReduceMemoryUsageStep(),
        #FeatureEngineeringLagStep(lags=[1,2,3,5,11,23], columns=["tn", "cust_request_qty", "stock_final"]),
        FeatureEngineeringLagStep(lags=[1,2,3,5,11], columns=["tn", "cust_request_qty", "stock_final"]),
        RollingMeanFeatureStep(window=3, columns=["tn", "cust_request_qty", "stock_final"]),
        RollingMaxFeatureStep(window=3, columns=["tn", "cust_request_qty", "stock_final"]),
        RollingMinFeatureStep(window=3, columns=["tn", "cust_request_qty", "stock_final"]),
        RollingMeanFeatureStep(window=9, columns=["tn", "cust_request_qty", "stock_final"]),
        RollingMaxFeatureStep(window=9, columns=["tn", "cust_request_qty", "stock_final"]),
        RollingMinFeatureStep(window=9, columns=["tn", "cust_request_qty", "stock_final"]),

        ReduceMemoryUsageStep(),

        RollingStdFeatureStep(window=3, columns=["tn", "cust_request_qty"]),
        #RollingStdFeatureStep(window=6, columns=["tn", "cust_request_qty"]),
        RollingStdFeatureStep(window=12, columns=["tn", "cust_request_qty"]), 

        RollingSkewFeatureStep(window=3, columns=["tn", "cust_request_qty"]),
        #RollingSkewFeatureStep(window=6, columns=["tn", "cust_request_qty"]),
        RollingSkewFeatureStep(window=12, columns=["tn", "cust_request_qty"]),
        ReduceMemoryUsageStep(),

        RollingZscoreFeatureStep(window=3, columns=["tn", "cust_request_qty"]),
        #RollingZscoreFeatureStep(window=6, columns=["tn", "cust_request_qty"]),
        RollingZscoreFeatureStep(window=12, columns=["tn", "cust_request_qty"]),
        DiffFeatureStep(periods=1, columns=["tn", "cust_request_qty", "stock_final"]),
        DiffFeatureStep(periods=2, columns=["tn", "cust_request_qty", "stock_final"]),
        DiffFeatureStep(periods=3, columns=["tn", "cust_request_qty", "stock_final"]),
        # DiffFeatureStep(periods=4, columns=["tn", "cust_request_qty", "stock_final"]),
        DiffFeatureStep(periods=5, columns=["tn", "cust_request_qty", "stock_final"]),
        DiffFeatureStep(periods=11, columns=["tn", "cust_request_qty", "stock_final"]),
        FeatureEngineeringProductCatInteractionStep(cat="cat1", tn="tn"),
        #FeatureEngineeringProductCatInteractionStep(cat="cat2", tn="tn"),
        #FeatureEngineeringProductCatInteractionStep(cat="cat3", tn="tn"),
        #FeatureEngineeringProductCatInteractionStep(cat="product_id", tn="tn"),

        CreateTotalCategoryStep(cat="cat1"),
        CreateTotalCategoryStep(cat="cat2"),
        CreateTotalCategoryStep(cat="cat3"),
        CreateTotalCategoryStep(cat="brand"),
        CreateTotalCategoryStep(cat="customer_id"),
        CreateTotalCategoryStep(cat="product_id"),
                
        #CreateTotalCategoryStep(cat="cat1", tn="stock_final"),
        #CreateTotalCategoryStep(cat="cat2", tn="stock_final"),
        #CreateTotalCategoryStep(cat="cat3", tn="stock_final"),

        #ReduceMemoryUsageStep(),
        FeatureDivInteractionStep(columns=[
                ("tn", "tn_cat1_vendidas"), 
                ("tn", "tn_cat2_vendidas"), 
                ("tn", "tn_cat3_vendidas"), 
                ("tn", "tn_brand_vendidas")]
        ),
        #ReduceMemoryUsageStep(),

        FeatureProdInteractionStep(columns=[("tn", "cust_request_qty")]),
        CreateWeightByCustomerStep(),
        CreateWeightByProductStep(),
        TimeDecayWeghtedProductIdStep(),
        ReduceMemoryUsageStep(),
        DeleteBadColumns(),
        SaveDataFrameStep(df_name="df", file_name="df_fe_big_exp.pickle"),
        PrepareXYStep(),
        TrainModelStep(params={"num_leaves":31, "feature_fraction":0.5}),
        PredictStep(),
        InverseResidualTargetStep(),
        #InverseLog1pTranformation(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        EvaluatePredictionsSteps(),
        #PlotFeatureImportanceStep(),

    ],
    optimize_arftifacts_memory=True
)

class DropColumnsStep(PipelineStep):
    def __init__(self, columns: list, name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns

    def execute(self, df: pd.DataFrame) -> Dict:
        df.drop(columns=self.columns, inplace=True, errors='ignore')
        return {"df": df}
    

model_test_pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep(path="df_fe_big_exp.pickle"),
        DateRelatedFeaturesStep(),
        SplitDataFrameStep2(df="df", test_date=29, gap=1),
        #CreateResidualTargetStep(target_col="tn", window=12),
        #CreateMultiDiffTargetColumStep(target_col=tn),
        #CreateTargetColumDiffStep(target_col="tn"),
        CreateTargetColumStep(target_col="tn"),
        ReduceMemoryUsageStep(),
        #DropColumnsStep(columns=["weight"]),
        PrepareXYStep(),
        #TrainModelStep(model_cls=XGBOOSTPipelineModel, params=xgb_params),
        #TrainModelStep(model_cls=MLPPipelineModel),
        TrainModelStep(params={"num_leaves":31}),
        #TrainNNModelStep(model_cls=NNPipelineModel),
        #TrainModelStep(folds=0, params={'num_leaves': 31, "n_estimators": 200, "learning_rate": 0.1}),
        PredictStep(),
        #PredictNNModelStep(),
        #InverseResidualTargetStep(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        EvaluatePredictionsSteps(),
        EvaluatePredictionsOptimizatedSteps(), 
        #PlotFeatureImportanceStep(),
        #SaveExperimentStep(exp_name="test_model",),
    ]
)
#pipeline.run()
model_test_pipeline.run()