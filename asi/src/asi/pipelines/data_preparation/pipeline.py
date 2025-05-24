
from kedro.pipeline import node, Pipeline
from .nodes import (
    preprocess_recommendations,
    create_user_game_features,
    prepare_training_data
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            preprocess_recommendations,
            inputs="recommendations",
            outputs="filtered_recommendations",
            name="preprocess_recommendations"
        ),
        node(
            create_user_game_features,
            inputs=["filtered_recommendations", "games"],
            outputs="enriched_features",
            name="create_user_game_features"
        ),
        node(
            prepare_training_data,
            inputs="enriched_features",
            outputs="training_data",
            name="prepare_training_data"
        ),
    ])