from pipeline import Pipeline
from pipeline.steps import *
from pipeline.steps.models.nn import NNPipelineModel, TrainNNModelStep, PredictNNModelStep


class PipelineComposition:
    def __init__(self, pipelines: Dict[Pipeline, List[Pipeline]]):
        self.pipelines = pipelines # los pipelines son grafos

    def run(self):
        for pipeline, dependencies in self.pipelines.items():
            needs = pipeline.needs
            if needs and not dependencies:
                raise ValueError(f"Pipeline '{pipeline.name}' has dependencies but no previous pipeline to provide them.")
            if dependencies:
                for dep in dependencies:
                    dep.run()
                    for need in needs:
                        artifact = dep.get_artifact(need, raise_not_found=False)
                        if artifact is not None:
                            pipeline.save_artifact(need, artifact)
            pipeline.run()



tn = "tn"

xgb_params = {
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbose": 0,
}

model_test_pipeline = Pipeline(
    #needs=["df"],
    steps=[
        LoadDataFrameFromPickleStep(path="df_tsfresh_base.pickle"),
        SplitDataFrameStep(df="df", test_date="2019-10", gap=1),
        CreateTargetColumStep(target_col=tn),
        #CreateMultiDiffTargetColumStep(target_col=tn),
        #CreateTargetColumDiffStep(target_col=tn),
        ScaleFeatureStep(column=".*tn.*", override=False, regex=True),  
        ScaleFeatureStep(column="target", override=True, scaler=PipelineMinMaxScaler),
        ReduceMemoryUsageStep(),
        PrepareXYStep(),
        #TrainModelStep(model_cls=XGBOOSTPipelineModel, params=xgb_params),
        #TrainModelStep(model_cls=MLPPipelineModel),
        TrainModelStep(folds=0, params={'num_leaves': 31, "n_estimators": 1200, "learning_rate": 0.01}),
        #TrainNNModelStep(model_cls=NNPipelineModel),
        #TrainModelStep(folds=0, params={'num_leaves': 31, "n_estimators": 200, "learning_rate": 0.1}),
        PredictStep(),
        #PredictNNModelStep(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        EvaluatePredictionsSteps(),
        PlotFeatureImportanceStep(),
        #SaveExperimentStep(exp_name="test_model",),
    ]
)

model_kaggle_pipeline = Pipeline(
    needs=["total_error"],
    steps=[
        LoadDataFrameFromPickleStep(path="df_fe_reducido.pickle"),
        SplitDataFrameStep(df="df", test_date="2019-12", gap=1),
        CreateTargetColumStep(target_col=tn),
        #CreateMultiDiffTargetColumStep(target_col=tn),
        #CreateTargetColumDiffStep(target_col=tn),
        ScaleFeatureStep(column=".*tn.*", override=False, regex=True),
        ScaleFeatureStep(column="target", override=True, scaler=PipelineStandarScaler),
        ReduceMemoryUsageStep(),

        PrepareXYStep(),
        TrainModelStep(model_cls=XGBOOSTPipelineModel, params=xgb_params),

        #TrainModelStep(folds=0, params={"n_estimators": 800}),
        #TrainModelStep(folds=0, params={'num_leaves': 31, "n_estimators": 200, "learning_rate": 0.1}),

        PredictStep(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        PlotFeatureImportanceStep(),
        KaggleSubmissionStep(),
        SaveSubmissionStep(exp_name="test_new_pipeline_xgboost_standard_scaler_target_tn_scaled"),
    ]
)


pipeline_fe_big = Pipeline(
    steps=[
        LoadDataFrameStep(path="df_intermedio.parquet"),
        #FilterProductForTestingStep(total_products_ids=10, random=False),
         FilterProductsIDStep(dfs=["df"]),
        # SplitDataFrameStep(df="df", test_date="2019-12", gap=1),
        # ScaleFeatureStep(column="tn"),

        # ScaleFeatureStep(column="cust_request_qty"),
        # ScaleFeatureStep(column="cust_request_tn"),
        CreateSerieIdStep(),
        DateRelatedFeaturesStep(),
        CastDataTypesStep(dtypes=
        {
            "product_id": "uint32",
            "customer_id": "uint32",
            "mes": "uint16",
            "year": "uint16",
            "brand": "category",
            "cat1": "category",
            "cat2": "category",
            "cat3": "category",
        }
        ),
        # ReduceMemoryUsageStep(),
        FeatureEngineeringLagStep(lags=[1, 2, 3,4,5,6, 11,20,30], columns=[tn, "cust_request_qty"]),
        RollingMeanFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingMaxFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingMinFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingMeanFeatureStep(window=6, columns=[tn, "cust_request_qty"]),
        RollingMaxFeatureStep(window=6, columns=[tn, "cust_request_qty"]),
        RollingMinFeatureStep(window=6, columns=[tn, "cust_request_qty"]),
        RollingMeanFeatureStep(window=12, columns=[tn, "cust_request_qty"]),
        RollingMaxFeatureStep(window=12, columns=[tn, "cust_request_qty"]),
        RollingMinFeatureStep(window=12, columns=[tn, "cust_request_qty"]),
        ReduceMemoryUsageStep(),


        RollingStdFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingStdFeatureStep(window=6, columns=[tn, "cust_request_qty"]),
        RollingStdFeatureStep(window=12, columns=[tn, "cust_request_qty"]), 

        RollingSkewFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingSkewFeatureStep(window=6, columns=[tn, "cust_request_qty"]),
        RollingSkewFeatureStep(window=12, columns=[tn, "cust_request_qty"]),
        ReduceMemoryUsageStep(),

        #RollingKurtosisFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        #RollingKurtosisFeatureStep(window=6, columns=[tn, "cust_request_qty"]), 
        #RollingKurtosisFeatureStep(window=12, columns=[tn, "cust_request_qty"]),

        RollingZscoreFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingZscoreFeatureStep(window=6, columns=[tn, "cust_request_qty"]),
        RollingZscoreFeatureStep(window=12, columns=[tn, "cust_request_qty"]),

        RollingAutocorrelationFeatureStep(window=3, lags=[3], columns=[tn, "cust_request_qty"]),

        # ReduceMemoryUsageStep(),

        DiffFeatureStep(periods=1, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=2, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=3, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=4, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=5, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=11, columns=[tn, "cust_request_qty"]),
         ReduceMemoryUsageStep(),

        CreateTotalCategoryStep(cat="cat1"),
        CreateTotalCategoryStep(cat="cat2"),
        CreateTotalCategoryStep(cat="cat3"),
        CreateTotalCategoryStep(cat="brand"),
        # ReduceMemoryUsageStep(),

        CreateTotalCategoryStep(cat="customer_id", tn=tn),
        CreateTotalCategoryStep(cat="product_id", tn=tn),
        # ReduceMemoryUsageStep(),

        CreateWeightByCustomerStep(),
        # ReduceMemoryUsageStep(),

        CreateWeightByProductStep(),
        ReduceMemoryUsageStep(),

        FeatureDivInteractionStep(columns=[
            ("tn", "tn_cat1_vendidas"), 
            ("tn", "tn_cat2_vendidas"), 
            ("tn", "tn_cat3_vendidas"), 
            ("tn", "tn_brand_vendidas")]
        ),
        #ReduceMemoryUsageStep(),

        FeatureProdInteractionStep(columns=[(tn, "cust_request_qty")]),
        ReduceMemoryUsageStep(),


        # steps para entrenar el modelo
        SaveDataFrameStep(df_name="df", file_name="df_fe_big.pickle")

    ],
    optimize_arftifacts_memory=True
)

model_test_grande = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep(path="df_fe_big.pickle"),
        DateRelatedFeaturesStep(),
        TimeDecayWeghtedProductIdStep(),
        ReduceMemoryUsageStep(),
        SplitDataFrameStep(df="df", test_date="2019-10", gap=1),
        CreateTargetColumStep(target_col="tn"),
        ScaleFeatureStep(column=".*tn.*", override=False, regex=True),  
        ScaleFeatureStep(column="target", override=True, scaler=PipelineStandarScaler),
        ReduceMemoryUsageStep(),
        PrepareXYStep(),
        TrainModelStep(folds=0, params={"max_bin":1024, 'num_leaves': 31, "n_estimators": 700, "learning_rate": 0.01}),
        PredictStep(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        EvaluatePredictionsSteps(),
        PlotFeatureImportanceStep(),
        LoadDataFrameFromPickleStep(path="df_fe_reducido.pickle"),
        SplitDataFrameStep(df="df", test_date="2019-12", gap=1),
        CreateTargetColumStep(target_col=tn),
        ScaleFeatureStep(column=".*tn.*", override=False, regex=True),
        ScaleFeatureStep(column="target", override=True, scaler=PipelineStandarScaler),
        ReduceMemoryUsageStep(),

        PrepareXYStep(),
        TrainModelStep(folds=0, params={"max_bin":1024, 'num_leaves': 31, "n_estimators": 700, "learning_rate": 0.01}),
        PredictStep(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        PlotFeatureImportanceStep(),
        KaggleSubmissionStep(),
        SaveSubmissionStep(exp_name="test_new_pipeline_150features"),


    ]
)


pipeline_fe = Pipeline(
    steps=[
        LoadDataFrameStep(path="df_intermedio.parquet"),
        FilterProductForTestingStep(total_products_ids=10, random=False),
        #FilterProductsIDStep(dfs=["df"]),
        #SplitDataFrameStep(df="df", test_date="2019-12", gap=1),
        #ScaleFeatureStep(column="tn"),

        #ScaleFeatureStep(column="cust_request_qty"),
        #ScaleFeatureStep(column="cust_request_tn"),
        CreateSerieIdStep(),
        DateRelatedFeaturesStep(),
        CastDataTypesStep(dtypes=
            {
                "product_id": "uint32", 
                "customer_id": "uint32",
                "mes": "uint16",
                "year": "uint16",
                "brand": "category",
                "cat1": "category",
                "cat2": "category",
                "cat3": "category",
            }
        ),
        #ReduceMemoryUsageStep(),
        FeatureEngineeringLagStep(lags=[1,2,3,5,11], columns=[tn, "cust_request_qty"]),
        RollingMeanFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingMaxFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        RollingMinFeatureStep(window=3, columns=[tn, "cust_request_qty"]),
        #ReduceMemoryUsageStep(),

        DiffFeatureStep(periods=1, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=2, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=5, columns=[tn, "cust_request_qty"]),
        DiffFeatureStep(periods=11, columns=[tn, "cust_request_qty"]),
        #ReduceMemoryUsageStep(),

        CreateTotalCategoryStep(cat="cat1"),
        CreateTotalCategoryStep(cat="cat2"),
        CreateTotalCategoryStep(cat="cat3"),
        CreateTotalCategoryStep(cat="brand"),
        #ReduceMemoryUsageStep(),

        CreateTotalCategoryStep(cat="customer_id", tn=tn),
        CreateTotalCategoryStep(cat="product_id", tn=tn),
        #ReduceMemoryUsageStep(),
 
        FeatureDivInteractionStep(columns=[
            ("tn", "tn_cat1_vendidas"), 
            ("tn", "tn_cat2_vendidas"), 
            ("tn", "tn_cat3_vendidas"), 
            ("tn", "tn_brand_vendidas")]
        ),
        #ReduceMemoryUsageStep(),

        FeatureProdInteractionStep(columns=[(tn, "cust_request_qty")]),
        CreateWeightByCustomerStep(),
        #ReduceMemoryUsageStep(),

        CreateWeightByProductStep(),
        ReduceMemoryUsageStep(),


        # steps para entrenar el modelo
        SaveDataFrameStep(df_name="df", file_name="df_fe_exp.pickle")


    ],
    optimize_arftifacts_memory=True
)

pipeline = PipelineComposition(
    pipelines={
        pipeline_fe: [],
        model_test_pipeline: [pipeline_fe],
        model_kaggle_pipeline: [pipeline_fe, model_test_pipeline]
    }
)

pipeline_no_fe = PipelineComposition(
    pipelines={
        #model_test_pipeline: [],
        model_kaggle_pipeline: [model_test_pipeline]
    }
)
#pipeline_nn = PipelineComposition(
#    pipelines={
#        model_test_pipeline: [pipeline_fe_for_nn],
#    }
#)
#pipeline_no_fe.run()
#pipeline_nn.run()
#pipeline_fe_big.run()
#pipeline_fe_big.run()
model_test_grande.run()
