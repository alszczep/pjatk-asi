# from __future__ import annotations
from asi.src.asi.pipelines.data_preparation import create_pipeline
# from pipelines.data_preparation import create_pipeline

def register_pipelines():
 return {"data_processing": create_pipeline()}
