from kedro.pipeline import node, Pipeline, pipeline
from asi.src.asi.pipelines.data_preparation.nodes import merge_datasets

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=merge_datasets,
            inputs=["recommendations", "games"],
            outputs="joined_recommendations_games",
            name="merge_datasets_node"
        ),
    ])
