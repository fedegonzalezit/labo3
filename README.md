# Pipeline System Documentation

This documentation will help you understand how to create and run your own data processing experiments using our pipeline system. The system is designed to be flexible and easy to use.

## What is a Pipeline?

A pipeline is a sequence of steps that process your data in a specific order. Each step performs a specific task, like loading data, transforming it, or training a model. The pipeline system helps you organize these steps and manage the data that flows between them.

## Basic Concepts

### Steps
- Steps are the building blocks of your pipeline
- Each step performs one specific task
- Steps can pass data to other steps through "artifacts"
- Steps can be cached to avoid re-running them unnecessarily

### Artifacts
- Artifacts are pieces of data that steps create and share
- Examples: datasets, models, predictions, etc.
- Steps can save artifacts for other steps to use
- Artifacts are managed automatically by the pipeline

## Creating Your Own Experiment

Here's a simple example of how to create your own experiment:

```python
from pipeline import Pipeline
from pipeline.steps import *

# Create a pipeline with some steps
my_pipeline = Pipeline(
    steps=[
        LoadDataFrameStep(path="my_data.csv"),
        CreateTargetColumnStep(target_col="sales"),
        PrepareXYStep(),
        TrainModelStep(),
        PredictStep(),
        EvaluatePredictionsSteps()
    ]
)

# Run the pipeline
my_pipeline.run()
```

## Creating Your Own Steps

To create your own step, you need to:

1. Create a new class that inherits from `PipelineStep`
2. Implement the `execute` method with parameters that match the artifacts you want to use
3. Return a dictionary with the artifacts you want to save

Here's an example:

```python
from pipeline.base import PipelineStep

class MyCustomStep(PipelineStep):
    def __init__(self, window_size: int = 3):
        super().__init__()
        self.window_size = window_size

    def execute(self, input_data, previous_artifact=None, pipeline=None):
        # The parameters in execute() automatically receive artifacts from previous steps
        # pipeline is optional and can be used if you need direct access to the pipeline
        
        # Process the data
        result = self.process_data(input_data, previous_artifact)
        
        # Return a dictionary with the artifacts you want to save
        return {
            "processed_data": result,
            "metadata": {"window_size": self.window_size}
        }

    def process_data(self, data, previous_artifact):
        # Your processing logic here
        return processed_data
```

The key points about step creation:

1. **Parameter Names**: The parameter names in your `execute` method should match the artifact names you want to use from previous steps
2. **Return Values**: Return a dictionary where:
   - Keys are the names you want to give to your artifacts
   - Values are the actual data you want to save
3. **Pipeline Access**: The `pipeline` parameter is optional and will be automatically injected if you include it
4. **Artifact Management**: You don't need to manually save artifacts - the pipeline system handles this automatically based on your return values

Here's a more practical example:

```python
class RollingMeanFeatureStep(PipelineStep):
    def __init__(self, window: int = 3, columns: List[str] = None):
        super().__init__()
        self.window = window
        self.columns = columns or []

    def execute(self, df, pipeline=None):
        # df is automatically received from a previous step that saved an artifact named "df"
        
        # Calculate rolling means
        for col in self.columns:
            df[f"{col}_rolling_mean"] = df[col].rolling(window=self.window).mean()
        
        # Return the modified dataframe
        return {"df": df}
```

## Using Caching

The pipeline system includes two types of caching:

1. In-memory caching: Stores results in RAM
2. Disk caching: Stores results on your hard drive

To use caching, simply add it to your step:

```python
# In-memory caching
step = MyCustomStep(parameter1, parameter2).in_memory_cache()

# Disk caching
step = MyCustomStep(parameter1, parameter2).in_disk_cache(cache_dir=".cache")
```

## Pipeline Composition

You can create complex experiments by combining multiple pipelines:

```python
from pipeline import PipelineComposition

# Create multiple pipelines
pipeline1 = Pipeline(steps=[...])
pipeline2 = Pipeline(steps=[...])

# Combine them
composition = PipelineComposition(
    pipelines={
        pipeline1: [],  # No dependencies
        pipeline2: [pipeline1]  # Depends on pipeline1
    }
)

# Run the composition
composition.run()
```

## Best Practices

1. **Step Organization**
   - Keep steps focused on one task
   - Use clear, descriptive names
   - Document what each step does

2. **Data Management**
   - Use meaningful artifact names
   - Clean up artifacts when they're no longer needed
   - Use caching for expensive operations

3. **Error Handling**
   - Add proper error messages
   - Validate inputs and outputs
   - Use try-except blocks when appropriate

## Example: Complete Experiment

Here's a complete example of a data analysis experiment:

```python
from pipeline import Pipeline
from pipeline.steps import *

# Create the pipeline
experiment = Pipeline(
    steps=[
        # Load and prepare data
        LoadDataFrameStep(path="sales_data.csv"),
        CreateTargetColumnStep(target_col="revenue"),
        
        # Feature engineering
        DateRelatedFeaturesStep(),
        RollingMeanFeatureStep(window=3, columns=["revenue"]),
        
        # Model training and evaluation
        PrepareXYStep(),
        TrainModelStep(),
        PredictStep(),
        EvaluatePredictionsSteps(),
        
        # Save results
        SaveExperimentStep(exp_name="my_experiment")
    ]
)

# Run the experiment
experiment.run()
```

## Need Help?

If you need help creating your own steps or experiments, you can:

1. Look at the existing steps in the `pipeline/steps` directory
2. Check the example experiments in the `experiments` directory
3. Review the base pipeline implementation in `pipeline/base.py`

Remember that the pipeline system is designed to be flexible and extensible. Don't hesitate to create your own steps and experiments! 