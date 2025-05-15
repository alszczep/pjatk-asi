from kedro.pipeline import node, Pipeline, pipeline
from .nodes import preprocess_recommendations, enrich_with_game_info

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            preprocess_recommendations,
            inputs="recommendations",
            outputs="cleaned_recommendations",
            name="preprocess_recommendations"
        ),
        node(
            enrich_with_game_info,
            inputs=["cleaned_recommendations", "games"],
            outputs="recommendations_enriched",
            name="enrich_with_game_info"
        ),
    ])
