"""Project pipelines."""
from .pipelines import data_preparation

# from pipelines.data_preparation import create_pipeline

def register_pipelines():
    return {"data_processing": data_preparation.create_pipeline()}

