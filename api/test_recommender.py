import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def generate_game_recommendations(
        model_path: str,
        all_features: pd.DataFrame,
        user_liked_games: list,
        top_n: int = 10,
        min_reviews: int = 50
) -> pd.DataFrame:
    try:
        predictor = TabularPredictor.load(model_path)
    except Exception:
        return pd.DataFrame()
    games_df = all_features.groupby('app_id').agg({
        'title': 'first',
        'positive_ratio': 'first',
        'user_reviews': 'first',
        'price_final': 'first',
        **{col: 'first' for col in all_features.columns
           if col not in ['app_id', 'title', 'positive_ratio', 'user_reviews', 'price_final']}
    }).reset_index()

    if 'user_reviews' in games_df.columns:
        games_df = games_df[games_df['user_reviews'] >= min_reviews]

    liked_games_df = games_df[games_df['app_id'].isin(user_liked_games)]
    if len(liked_games_df) == 0:
        return pd.DataFrame()

    candidate_games = games_df[~games_df['app_id'].isin(user_liked_games)].copy()
    if len(candidate_games) == 0:
        return pd.DataFrame()

    user_profile = _create_user_profile(liked_games_df)
    ml_scores = _get_ml_predictions(predictor, candidate_games, user_profile)
    content_scores = _calculate_content_similarity(liked_games_df, candidate_games)
    quality_scores = _calculate_quality_scores(candidate_games)
    final_scores = (
            ml_scores * 0.5 +
            content_scores * 0.3 +
            quality_scores * 0.2
    )

    recommendations = pd.DataFrame({
        'app_id': candidate_games['app_id'].values,
        'title': candidate_games['title'].values,
        'recommendation_score': final_scores,
        'user_reviews': candidate_games['user_reviews'].values,
        'positive_ratio': candidate_games['positive_ratio'].values,
        'price_final': candidate_games['price_final'].values
    })

    recommendations = recommendations.sort_values('recommendation_score', ascending=False).head(top_n)
    recommendations['rank'] = range(1, len(recommendations) + 1)

    return recommendations.reset_index(drop=True)


def _create_user_profile(liked_games_df):
    numeric_cols = liked_games_df.select_dtypes(include=[np.number]).columns
    return liked_games_df[numeric_cols].mean()

def _get_ml_predictions(predictor, candidate_games, user_profile):
    model_features = [col for col in predictor.feature_metadata.get_features()
                      if col != 'is_recommended']

    prediction_data = candidate_games.copy()

    for feature, value in user_profile.items():
        if feature in model_features and feature in prediction_data.columns:
            prediction_data[feature] = value

    if 'hours' in model_features:
        base_hours = user_profile.get('user_avg_hours', 10.0)
        game_factor = candidate_games.get('game_avg_hours', base_hours) / base_hours
        prediction_data['hours'] = np.maximum(0.5, base_hours * game_factor * np.random.lognormal(0, 0.3,
                                                                                                  len(candidate_games)))
    if 'helpful' in model_features:
        base_helpful = user_profile.get('user_helpful_rate', 0.1) * 10
        prediction_data['helpful'] = np.random.poisson(base_helpful, len(candidate_games))

    if 'funny' in model_features:
        base_funny = user_profile.get('user_funny_rate', 0.05) * 10
        prediction_data['funny'] = np.random.poisson(base_funny, len(candidate_games))

    if 'hours_vs_user_avg' in model_features and 'user_avg_hours' in prediction_data.columns:
        prediction_data['hours_vs_user_avg'] = prediction_data['hours'] / (prediction_data['user_avg_hours'] + 1e-6)

    if 'hours_vs_game_avg' in model_features and 'game_avg_hours' in prediction_data.columns:
        prediction_data['hours_vs_game_avg'] = prediction_data['hours'] / (prediction_data['game_avg_hours'] + 1e-6)

    if 'price_vs_user_avg' in model_features and 'user_avg_price_paid' in prediction_data.columns:
        prediction_data['price_vs_user_avg'] = prediction_data['price_final'] / (
                    prediction_data['user_avg_price_paid'] + 1e-6)

    available_features = [col for col in model_features if col in prediction_data.columns]
    model_input = prediction_data[available_features].fillna(0)

    try:
        predictions = predictor.predict_proba(model_input)

        if hasattr(predictions, 'iloc') and predictions.shape[1] > 1:
            scores = predictions.iloc[:, 1].values
        else:
            scores = predictions if isinstance(predictions, np.ndarray) else predictions.values

        scores = np.clip(scores, 0, 1)

        return scores

    except Exception:
        return np.random.uniform(0.3, 0.7, len(candidate_games))


def _calculate_content_similarity(liked_games_df, candidate_games):
    feature_cols = ['positive_ratio', 'user_reviews', 'price_final']
    possible_features = [
        'game_avg_hours', 'game_rec_rate', 'win', 'mac', 'linux',
        'game_positive_ratio', 'game_quality_score', 'is_popular_game'
    ]

    for feat in possible_features:
        if feat in liked_games_df.columns and feat in candidate_games.columns:
            feature_cols.append(feat)

    available_features = [col for col in feature_cols
                          if col in liked_games_df.columns and col in candidate_games.columns]

    if not available_features:
        return np.zeros(len(candidate_games))

    liked_features = liked_games_df[available_features].fillna(0)
    candidate_features = candidate_games[available_features].fillna(0)

    if len(liked_features) == 0:
        return np.zeros(len(candidate_games))

    scaler = StandardScaler()
    all_features = pd.concat([liked_features, candidate_features])
    all_features_scaled = scaler.fit_transform(all_features)

    liked_scaled = all_features_scaled[:len(liked_features)]
    candidate_scaled = all_features_scaled[len(liked_features):]
    user_profile = liked_scaled.mean(axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_profile, candidate_scaled)[0]
    similarities = (similarities + 1) / 2

    return similarities


def _calculate_quality_scores(candidate_games):

    scores = np.zeros(len(candidate_games))
    if 'positive_ratio' in candidate_games.columns:
        scores += candidate_games['positive_ratio'].fillna(50) / 100 * 0.4
    if 'user_reviews' in candidate_games.columns:
        review_scores = np.log1p(candidate_games['user_reviews'].fillna(1)) / 10
        review_scores = np.clip(review_scores, 0, 1)
        scores += review_scores * 0.3
    if 'game_rec_rate' in candidate_games.columns:
        scores += candidate_games['game_rec_rate'].fillna(0.5) * 0.3

    return np.clip(scores, 0, 1)


def get_recommendations(model_path: str, data_path: str, liked_games: list, num_recommendations: int = 10):
    try:
        # Load data
        enriched_data = pd.read_parquet(data_path)

        # Generate recommendations
        recommendations = generate_game_recommendations(
            model_path=model_path,
            all_features=enriched_data,
            user_liked_games=liked_games,
            top_n=num_recommendations,
            min_reviews=100
        )

        if recommendations.empty:
            return None

        return recommendations

    except Exception:
        return None


if __name__ == "__main__":
    # Example usage
    model_path = "data/06_models/autogluon_recommender"
    data_path = "data/04_feature/enriched_features.parquet"
    liked_games = [730, 7940, 440]  # MUST HAVE TWO GAMES AT LEAST!!!!

    recommendations = get_recommendations(model_path, data_path, liked_games, 5)

    for rec in recommendations:
        print(f"{rec['rank']}. {rec['title']} (Score: {rec['recommendation_score']:.3f})")