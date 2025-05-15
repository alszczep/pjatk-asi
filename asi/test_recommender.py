import pandas as pd
from autogluon.tabular import TabularPredictor

def recommend_similar_games(model_path: str, games_df: pd.DataFrame, liked_app_ids: list[int], top_n: int = 10) -> pd.DataFrame:
    predictor = TabularPredictor.load(model_path)

    all_candidates = (
        games_df[~games_df["app_id"].isin(liked_app_ids)]
        .drop_duplicates(subset="app_id")  # ensures each game appears only once
        .copy()
    )

    drop_cols = ["user_id", "app_id", "title", "is_recommended", "date", "review_id"]
    all_candidates = all_candidates.drop(columns=[col for col in drop_cols if col in all_candidates.columns])

    all_candidates = all_candidates.fillna(-1)

    proba_df = predictor.predict_proba(all_candidates)
    scores = proba_df[True]  # Probabilities of being recommended

    top_indices = scores.sort_values(ascending=False).head(top_n).index
    top_games = games_df.loc[top_indices][["app_id", "title"]].copy()
    top_games["score"] = scores.loc[top_indices].values

    return top_games.reset_index(drop=True)

# Example usage
if __name__ == "__main__":
    # Paths
    MODEL_PATH = "data/06_models/automl_model"
    GAMES_PATH = "data/04_feature/recommendations_enriched.parquet"

    # Load games dataset
    games = pd.read_parquet(GAMES_PATH)

    # User likes these app_ids
    liked_app_ids = [730, 289070, 1086940]  # CS:GO, Civ VI, Baldur's Gate 3

    # Get recommendations
    top_games = recommend_similar_games(MODEL_PATH, games, liked_app_ids, top_n=10)

    print("\nTop Recommended Games:")
    print(top_games)
