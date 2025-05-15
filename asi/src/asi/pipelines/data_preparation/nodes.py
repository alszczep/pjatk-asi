import pandas as pd
def preprocess_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    recommendations['date'] = pd.to_datetime(recommendations['date'])
    recommendations['is_recommended'] = recommendations['is_recommended'].astype(bool)

    filtered = recommendations.groupby('user_id').filter(lambda x: len(x) > 3)
    return filtered

def enrich_with_game_info(recommendations: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    return recommendations.merge(games, on="app_id", how="left")

