from pipeline import Pipeline
from pipeline.steps import (
    LoadDataFrameFromPickleStep,
    SplitDataFrameStep,
    CreateTargetColumStep,
    ScaleFeatureStep,
    ReduceMemoryUsageStep,
    PrepareXYStep,
    TrainModelStep,
    PredictStep,
    InverseScalePredictionsStep,
    IntegratePredictionsStep,
    EvaluatePredictionsSteps,
    PlotFeatureImportanceStep,
    KaggleSubmissionStep,
    SaveSubmissionStep,
    PipelineMinMaxScaler,
    FilterFirstDateStep
)
from pipeline.steps.df_setup import CreateTargetColumDiffStep
from pipeline.steps.scaler import PipelineStandarScaler
from pipeline import PipelineStep
import numpy as np

class Log1pStep(PipelineStep):
    def __init__(self, name = None, column="tn"):
        super().__init__(name)
        self.column = column

    def execute(self, df):
        grouped = df.groupby(["customer_id", "product_id"])
        df[f"{self.column}_log1p"] = grouped[self.column].transform(lambda x: x + 1).apply(np.log1p)
        return df
    
    

model_test_grande = Pipeline(
    steps=[
        #LoadDataFrameFromPickleStep(path="df_fe_big.pickle"),
        ##FilterFirstDateStep(first_date="2018-01-01"),
        #SplitDataFrameStep(df="df", test_date="2019-10", gap=1),
        #CreateTargetColumDiffStep(target_col="tn"),
        #ScaleFeatureStep(column="target", override=True, scaler=PipelineStandarScaler),
        ##CreateTargetColumStep(target_col="tn"),
        ##ScaleFeatureStep(column="target", override=True, scaler=PipelineMinMaxScaler),
        ##ScaleFeatureStep(column=".*tn.*", override=False, regex=True),  
        #ReduceMemoryUsageStep(),
        #PrepareXYStep(),
        #TrainModelStep(folds=0, params={"max_bin":1024, 'num_leaves': 256, "n_estimators": 700, "learning_rate": 0.01}),
        #PredictStep(),
        #InverseScalePredictionsStep(),
        #IntegratePredictionsStep(),
        #EvaluatePredictionsSteps(),
        #PlotFeatureImportanceStep(),

        LoadDataFrameFromPickleStep(path="df_fe_big.pickle"),
        #FilterFirstDateStep(first_date="2018-01-01"),
        SplitDataFrameStep(df="df", test_date="2019-12", gap=1),
        CreateTargetColumDiffStep(target_col="tn"),
        #ScaleFeatureStep(column=".*tn.*", override=False, regex=True),  
        ScaleFeatureStep(column="target", override=True, scaler=PipelineStandarScaler),
        ReduceMemoryUsageStep(),
        PrepareXYStep(),
        TrainModelStep(folds=0, params={"max_bin":1024, 'num_leaves': 256, "n_estimators": 700, "learning_rate": 0.01}),
        PredictStep(),
        InverseScalePredictionsStep(),
        IntegratePredictionsStep(),
        PlotFeatureImportanceStep(),
        KaggleSubmissionStep(),
        SaveSubmissionStep(exp_name="test_new_pipeline_100features"),
    ]
)

model_test_grande.run()
