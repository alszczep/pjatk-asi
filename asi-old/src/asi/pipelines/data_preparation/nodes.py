import pandas as pd

def merge_datasets(recommendations: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
 return recommendations.join(games, on="app_id")