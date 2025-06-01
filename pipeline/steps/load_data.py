from pipeline import PipelineStep
import pandas as pd
from typing import Optional
import pickle

class LoadDataFrameStep(PipelineStep):
    """
    Example step that loads a DataFrame.
    """
    def __init__(self, path: str, name: Optional[str] = None):
        super().__init__(name)
        self.path = path

    def execute(self) -> None:
        df = pd.read_parquet(self.path)
        df = df.drop(columns=["periodo"], errors='ignore')
        return {"df": df}
    

class LoadScalerStep(PipelineStep):
    def __init__(self, artifact_name: str, file_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.file_name = file_name
        self.artifact_name = artifact_name

    def execute(self):
        """
        Carga un scaler previamente guardado desde un archivo.
        """
        with open(self.file_name, "rb") as f:
            scaler = pickle.load(f)
        return {self.artifact_name: scaler}   