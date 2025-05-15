"""Project pipelines."""
from kedro.pipeline import Pipeline
from .pipelines import data_preparation, modelling

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_processing": data_preparation.create_pipeline(),
        "modeling": modelling.create_pipeline(),
        "__default__": data_preparation.create_pipeline() + modelling.create_pipeline(),
    }

