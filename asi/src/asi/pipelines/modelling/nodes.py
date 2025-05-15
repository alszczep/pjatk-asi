import pandas as pd
import os
from sklearn.metrics import classification_report
from autogluon.tabular import TabularPredictor
def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Target value counts:\n", df["is_recommended"].value_counts())
    print("Number of rows before cleaning:", len(df))

    # Drop leak/non-feature columns
    drop_cols = ["user_id", "app_id", "date", "review_id", "title"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Drop rows with missing target
    df = df.dropna(subset=["is_recommended"])

    return df

def train_automl_model(df: pd.DataFrame) -> TabularPredictor:
    label = "is_recommended"

    print("Training on shape:", df.shape)
    model_path = os.path.abspath("data/06_models/automl_model")  # full path
    os.makedirs(model_path, exist_ok=True)  # ensure the directory exists

    df_sample = df.sample(n=2000, random_state=42)

    predictor = TabularPredictor(label=label, path=model_path).fit(
        df_sample,
        verbosity=4,
        raise_on_no_models_fitted=False,
        presets="best_quality",  # Or try "medium_quality_faster_train"
        ag_args_fit={"ag.max_memory_usage_ratio": 0.5},  # only use 50% of memory
        excluded_model_types=["NN_TORCH", "XGBOOST", "FASTAI"]  # disable heavy models
    )

    return model_path


def predict_recommendations(model_path: str, df):
    predictor = TabularPredictor.load(model_path)
    df = df.fillna(-1)
    return predictor.predict_proba(df)



def evaluate_model(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(model_path)
    y_true = df["is_recommended"]
    X = df.drop(columns=["is_recommended"])
    y_pred = predictor.predict(X)

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    print("=== Evaluation Summary ===")
    print(report_df)

    return report_df
