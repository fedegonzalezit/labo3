from pipeline import Pipeline
from pipeline.steps import *


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
                        artifact = dep.get_artifact(need)
                        if artifact is None:
                            raise ValueError(f"Artifact '{need}' not found in the previous pipeline '{dep.name}'.")
                        pipeline.save_artifact(need, artifact)
            pipeline.run()



tn = "tn"

model_test_pipeline = Pipeline(
    #needs=["df"],
    steps=[
        SplitDataFrameStep(df="df", test_date="2019-10", gap=1),
        #CreateTargetColumStep(target_col=tn),
        CreateTargetColumDiffStep(target_col=tn),
        #CreateMultiDiffTargetColumStep(target_col=tn),
        PrepareXYStep(),
        TrainModelStep(),
        PredictStep(),
        IntegratePredictionsStep(),
        InverseScalePredictionsStep(),
        EvaluatePredictionsSteps(),
        PlotFeatureImportanceStep(),
        #SaveExperimentStep(exp_name="test_model",),
    ]
)

model_kaggle_pipeline = Pipeline(
    needs=["total_error"],
    steps=[
        SplitDataFrameStep(df="df", test_date="2019-12", gap=0),
        #CreateTargetColumStep(target_col=tn),
        CreateTargetColumDiffStep(target_col=tn),
        #CreateMultiDiffTargetColumStep(target_col=tn),

        PrepareXYStep(),
        TrainModelStep(),
        PredictStep(),
        IntegratePredictionsStep(),
        InverseScalePredictionsStep(),
        PlotFeatureImportanceStep(),
        KaggleSubmissionStep(),
        SaveSubmissionStep(exp_name="test_new_pipeline"),
    ]
)



pipeline_fe = Pipeline(
    steps=[
        LoadDataFrameStep(path="df_intermedio.parquet"),
        #FilterProductForTestingStep(total_products_ids=10, random=False),
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

        CreateTotalCategoryStep(cat="cat1", tn=tn),
        CreateTotalCategoryStep(cat="cat2", tn=tn),
        CreateTotalCategoryStep(cat="cat3", tn=tn),
        #ReduceMemoryUsageStep(),

        CreateTotalCategoryStep(cat="brand", tn=tn),
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
        SaveDataFrameStep(df_name="df", file_name="df_fe")


    ],
    optimize_arftifacts_memory=True
)

#pipeline = PipelineComposition(
#    pipelines={
#        pipeline_fe: [],
#        model_test_pipeline: [pipeline_fe],
#        model_kaggle_pipeline: [pipeline_fe, model_test_pipeline]
#    }
#)

pipeline = PipelineComposition(
    pipelines={
        model_test_pipeline: [],
        model_kaggle_pipeline: [model_test_pipeline]
    }
)
pipeline.run()
