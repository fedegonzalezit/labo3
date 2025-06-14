from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple, Self
import time
import gc
import warnings
import os
import shutil
import pickle
from flaml import AutoML
import os
from glob import glob
import inspect
import logging
logger = logging.getLogger(__name__)        
import cloudpickle
import hashlib
import inspect
from typing import Any, Dict, Optional


SEED = 42
def fallback_latest_notebook():
    notebooks = glob("*.ipynb")
    if not notebooks:
        return None
    notebooks = sorted(notebooks, key=os.path.getmtime, reverse=True)
    return notebooks[0]



warnings.filterwarnings('ignore', category=FutureWarning)


class InDiskCacheWrapper:
    """
    Wrapper class to enable in-disk caching for pipeline steps.
    It uses the InDiskCache class to cache artifacts on disk.
    """
    def __init__(self, step: "PipelineStep", cache_dir: str = ".cache", execute_params: Optional[Dict[str, Any]] = None):
        self.step = step
        self.cache_dir = os.path.join(cache_dir, step.name)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._execute_params = execute_params or {}

    def execute(self, *args: Any, **kwargs: Any) -> None:
        """if the step has a cache, it hashes the parameters and checks if the result is already cached.
        note that params could be any object, so it uses cloudpickle to serialize them.
        If the result is cached, it returns the cached result.
        If not, it executes the step and saves the result in the cache.
        """
        # Bind args/kwargs to parameter names using original signature
        bound = inspect.signature(self.step.execute).bind(*args, **kwargs)
        #bound.apply_defaults()

        # also checks que values from __init__ for the hash
        init_params = self.step.__dict__.copy()
        # si los parametros con los que se inicializo cambiaron entonces deberia missear el cache
        bound.apply_defaults()

        # Serialize input arguments with cloudpickle
        try:
            serialized = cloudpickle.dumps(bound.arguments)
            # Include init parameters in the serialization
            serialized += cloudpickle.dumps(init_params)
        except Exception as e:
            raise ValueError(f"Failed to serialize for cache: {e}")

        # Generate a hash key from inputs
        hash_key = hashlib.sha256(serialized).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")

        # Load from cache or compute and save
        if os.path.exists(cache_file):
            print(f"Loading cached result for {self.step.name} from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            print(f"Cache miss for {self.step.name}, executing step and saving result to {cache_file}")
            result = self.step.execute(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

    def get_execute_params(self) -> Dict[str, Any]:
        """
        Get the parameters for the execute method of the wrapped step.
        """
        return self._execute_params
    
    @property
    def name(self) -> str:
        """
        Get the name of the step.
        """
        return self.step.name
    

class InMemoryCacheWrapper:
    """
    Wrapper class to enable in-memory caching for pipeline steps.
    It uses the InMemoryCache class to cache artifacts in memory.
    """
    cache = {}
    
    def __init__(self, step: "PipelineStep", execute_params: Optional[Dict[str, Any]] = None):
        self.step = step
        self._execute_params = execute_params or {}

    def execute(self, *args: Any, **kwargs: Any) -> None:
        """Execute the step and cache the result in memory."""
        # Bind args/kwargs to parameter names using original signature
        bound = inspect.signature(self.step.execute).bind(*args, **kwargs)

        init_params = self.step.__dict__.copy()
        # Merge init parameters with execute parameters
        bound.arguments.update(init_params)
        bound.apply_defaults()

        # Serialize input arguments with cloudpickle
        try:
            serialized = cloudpickle.dumps(bound.arguments)
        except Exception as e:
            raise ValueError(f"Failed to serialize for cache: {e}")

        # Generate a hash key from inputs
        hash_key = hashlib.sha256(serialized).hexdigest()

        # Load from cache or compute and save
        if hash_key in self.cache:
            print(f"Loading cached result for {self.step.name} from memory")
            return self.cache[hash_key]
        else:
            print(f"Cache miss for {self.step.name}, executing step and saving result in memory")
            result = self.step.execute(*args, **kwargs)
            self.cache[hash_key] = result
            return result

    def get_execute_params(self) -> Dict[str, Any]:
        """
        Get the parameters for the execute method of the wrapped step.
        """
        return self._execute_params
    
    @property
    def name(self) -> str:
        """
        Get the name of the step.
        """
        return self.step.name
    

class CachedPipelineMixin:
    def in_disk_cache(self, cache_dir: str = ".cache") -> Self:
        """
        It activate the in-disk cache using the InDisKCache class. returns the step itself.
        Args:
            cache_dir (str): Directory where the cache will be stored.
        """
        execute_params = self.get_execute_params()
        return InDiskCacheWrapper(self, cache_dir=cache_dir, execute_params=execute_params)
    
    def in_memory_cache(self) -> Self:
        """
        It activate the in-memory cache using the InMemoryCache class. returns the step itself.
        """
        execute_params = self.get_execute_params()
        return InMemoryCacheWrapper(self, execute_params=execute_params)
    

class PipelineStep(ABC, CachedPipelineMixin):
    """
    Abstract base class for pipeline steps.
    Each step in the pipeline must inherit from this class and implement the execute method.
    """
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a pipeline step.

        Args:
            name (str): Name of the step for identification and logging purposes.
        """
        self._name = name or self.__class__.__name__

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        Execute the pipeline step.
    
        Args:
            pipeline (Pipeline): The pipeline instance that contains this step.
        """
        pass

    def save_artifact(self, pipeline: "Pipeline", artifact_name: str, artifact: Any) -> None:
        """
        Save an artifact produced by this step to the pipeline.

        Args:
            pipeline (Pipeline): The pipeline instance.
            artifact_name (str): Name to identify the artifact.
            artifact (Any): The artifact to save.
        """
        pipeline.save_artifact(artifact_name, artifact)

    def get_artifact(self, pipeline: "Pipeline", artifact_name: str, default=None, raise_not_found=True) -> Any:
        """
        Retrieve a stored artifact from the pipeline.

        Args:
            pipeline (Pipeline): The pipeline instance.
            artifact_name (str): Name of the artifact to retrieve.
            default: Default value to return if the artifact is not found.
            raise_not_found (bool): Whether to raise an error if the artifact is not found.

        Returns:
            Any: The requested artifact or default value.
        """
        return pipeline.get_artifact(artifact_name, default=default, raise_not_found=raise_not_found)
    
    def del_artifact(self, pipeline: "Pipeline", artifact_name: str, soft=True) -> None:
        """
        Delete a stored artifact from the pipeline and free memory.

        Args:
            pipeline (Pipeline): The pipeline instance.
            artifact_name (str): Name of the artifact to delete.
            soft (bool): If True, performs a soft delete; if False, forces garbage collection.
        """
        pipeline.del_artifact(artifact_name, soft=soft)

    def get_execute_params(self):
        sig = inspect.signature(self.execute)
        return sig.parameters

        
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
    


class Pipeline:
    """
    Main pipeline class that manages the execution of steps and storage of artifacts.
    """
    def __init__(self, name=None, steps: Optional[List[PipelineStep]] = None, optimize_arftifacts_memory: bool = True, needs=None):
        """Initialize the pipeline."""
        self.steps: List[PipelineStep] = steps if steps is not None else []
        self.artifacts: Dict[str, Any] = {}
        self.optimize_arftifacts_memory = optimize_arftifacts_memory
        self.needs = needs or []
        self.finished = False
        self.name = name

    def add_step(self, step: PipelineStep, position: Optional[int] = None) -> None:
        """
        Add a new step to the pipeline.

        Args:
            step (PipelineStep): The step to add.
            position (Optional[int]): Position where to insert the step. If None, appends to the end.
        """
        if position is not None:
            self.steps.insert(position, step)
        else:
            self.steps.append(step)

    def save_artifact(self, artifact_name: str, artifact: Any) -> None:
        """
        Save an artifact from a given step.

        Args:
            artifact_name (str): Name to identify the artifact.
            artifact (Any): The artifact to save.
        """
        if not self.optimize_arftifacts_memory:
            self.artifacts[artifact_name] = artifact
        else:
            # guarda el artifact en /tmp/ para no guardarlo en memoria
            # si tiene self.name el path es /tmp/{self.name}/{artifact_name}
            if self.name:
                path = os.path.join("/tmp/", self.name)
            else:
                path = "/tmp/"
            if not os.path.exists(path):
                os.makedirs(path)
            artifact_path = os.path.join(path, artifact_name)
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
            self.artifacts[artifact_name] = artifact_path

    def get_artifact(self, artifact_name: str, default=None, raise_not_found=True) -> Any:
        """
        Retrieve a stored artifact.

        Args:
            artifact_name (str): Name of the artifact to retrieve.

        Returns:
            Any: The requested artifact.
        """
        if not self.optimize_arftifacts_memory:
            return self.artifacts.get(artifact_name)
        else:
            artifact_path = self.artifacts.get(artifact_name)
            if artifact_path and os.path.exists(artifact_path):
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
            else:
                if raise_not_found:
                    raise FileNotFoundError(f"Artifact {artifact_name} not found in /tmp/")
                return default
    
    def del_artifact(self, artifact_name: str, soft=True) -> None:
        """
        Delete a stored artifact and free memory.

        Args:
            artifact_name (str): Name of the artifact to delete.
        """
        del self.artifacts[artifact_name]
        if not soft:
            # Force garbage collection if not soft delete
            gc.collect()
    
    
    def run(self, verbose: bool = True, last_step_callback: Callable = None) -> None:
        """
        Execute all steps in sequence and log execution time.
        """        
        
        # Run steps from the last completed step
        if self.finished:
            if verbose:
                print("Pipeline has already finished. Skipping execution.")
            return
        
        for step in self.steps:
            if verbose:
                print(f"Executing step: {step.name}")
            start_time = time.time()
            params = self.__fill_params_from_step(step)
            artifacts_to_save = step.execute(**params)
            if artifacts_to_save is None:
                artifacts_to_save = {}
            self.__save_step_artifacts(artifacts_to_save)
            end_time = time.time()
            if verbose:
                print(f"Step {step.name} completed in {end_time - start_time:.2f} seconds")
        self.finished = True

    def __fill_params_from_step(self, step: PipelineStep) -> Dict[str, Any]:
        """
        Obtiene los nombres de los parametros de la implementacion de la funcion execute del paso. (excepto el pipeline el cual es obligatorio)
        luego obtengo todos los artefactos del pipeline y los paso como parametros al paso.
        """
        step_params = step.get_execute_params()
        params = {}
        for name, param in step_params.items():
            if name == 'pipeline':
                params[name] = self
            elif param.default is inspect.Parameter.empty:
                params[name] = self.get_artifact(name)
            else:
                params[name] = self.get_artifact(name, default=param.default, raise_not_found=False)
        return params

    

    def __save_step_artifacts(self, artifacts_to_save: Dict[str, Any]) -> None:
        """
        Save artifacts produced by a step to the pipeline.

        Args:
            artifacts_to_save (Dict[str, Any]): Artifacts to save.
        """

        for name, artifact in artifacts_to_save.items():
            self.save_artifact(name, artifact)



    def clear(self, collect_garbage: bool = False) -> None:
        """
        Clean up all artifacts and free memory.
        """
        if collect_garbage:
            del self.artifacts
            gc.collect()
        self.artifacts = {}
        self.last_step = None
        self.finished = False
