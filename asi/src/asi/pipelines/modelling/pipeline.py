from kedro.pipeline import Pipeline, node
from .nodes import (
    train_autogluon_recommender,
    improved_generate_game_recommendations,  # Updated function name
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
        # Optional: Add a recommendation generation node if you want it in the pipeline
        # Uncomment the lines below if you want to generate recommendations as part of the pipeline
        # node(
        #     improved_generate_game_recommendations,
        #     inputs=["autogluon_recommender_model", "enriched_features", "params:user_liked_games"],
        #     outputs="game_recommendations",
        #     name="generate_game_recommendations"
        # ),
    ])