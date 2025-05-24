from kedro.pipeline import Pipeline, node
from .nodes import (
    train_autogluon_recommender,
    evaluate_recommender
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            train_autogluon_recommender,
            inputs="training_data",
            outputs="autogluon_recommender_model",
            name="train_autogluon_recommender"
        ),
        node(
            evaluate_recommender,
            inputs=["autogluon_recommender_model", "training_data"],
            outputs="recommender_evaluation",
            name="evaluate_recommender"
        ),
    ])
