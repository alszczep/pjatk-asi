
from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_model_data,
    train_automl_model,
    evaluate_model,
    predict_recommendations
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            prepare_model_data,
            inputs="recommendations_enriched",
            outputs="model_input_table",
            name="prepare_model_data"
        ),
        node(
            train_automl_model,
            inputs="model_input_table",
            outputs="automl_model",
            name="train_model"
        ),
        node(
            evaluate_model,
            inputs=["automl_model", "model_input_table"],
            outputs="model_evaluation_report",
            name="evaluate_model"
        ),
        node(
            predict_recommendations,
            inputs=["automl_model", "model_input_table"],
            outputs="predictions",
            name="predict_recommendations"
        ),
    ])

