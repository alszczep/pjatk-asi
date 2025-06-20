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
        min_reviews: int = 50,
        diversity_factor: float = 0.1
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

    user_profile = _create_enhanced_user_profile(liked_games_df, all_features, user_liked_games)
    ml_scores = _get_ml_predictions(predictor, candidate_games, user_profile)
    content_scores = _calculate_content_similarity(liked_games_df, candidate_games)
    quality_scores = _calculate_quality_scores(candidate_games)
    popularity_penalty = _calculate_popularity_penalty(candidate_games, diversity_factor)

    final_scores = (
            ml_scores * 0.5 +
            content_scores * 0.25 +
            quality_scores * 0.2 +
            popularity_penalty * 0.05
    )

    recommendations = pd.DataFrame({
        'app_id': candidate_games['app_id'].values,
        'title': candidate_games['title'].values,
        'recommendation_score': final_scores,
        'ml_score': ml_scores,
        'content_score': content_scores,
        'quality_score': quality_scores,
        'user_reviews': candidate_games['user_reviews'].values,
        'positive_ratio': candidate_games['positive_ratio'].values,
        'price_final': candidate_games['price_final'].values
    })

    recommendations = recommendations.sort_values('recommendation_score', ascending=False).head(top_n)
    recommendations['rank'] = range(1, len(recommendations) + 1)

    return recommendations.reset_index(drop=True)


def _create_enhanced_user_profile(liked_games_df, all_features, user_liked_games):
    user_reviews = all_features[all_features['app_id'].isin(user_liked_games)]

    if len(user_reviews) > 0:
        user_profile = user_reviews.select_dtypes(include=[np.number]).mean()
    else:
        user_profile = liked_games_df.select_dtypes(include=[np.number]).mean()

    profile_dict = user_profile.to_dict()

    if len(user_reviews) > 0:
        profile_dict['user_avg_hours'] = user_reviews['hours'].mean()
        profile_dict['user_rec_rate'] = user_reviews['is_recommended'].mean()
        profile_dict['user_helpful_rate'] = user_reviews[
                                                'helpful'].mean() / 10 if 'helpful' in user_reviews.columns else 0.1
        profile_dict['user_funny_rate'] = user_reviews['funny'].mean() / 10 if 'funny' in user_reviews.columns else 0.05
        profile_dict['user_avg_price_paid'] = user_reviews['price_final'].mean()

    return profile_dict


def _get_ml_predictions(predictor, candidate_games, user_profile):
    model_features = [col for col in predictor.feature_metadata.get_features()
                      if col != 'is_recommended']

    prediction_data = candidate_games.copy()

    for feature, value in user_profile.items():
        if feature in model_features and feature in prediction_data.columns:
            prediction_data[feature] = value

    if 'hours' in model_features:
        base_hours = user_profile.get('user_avg_hours', 10.0)
        game_hours = candidate_games.get('game_avg_hours', base_hours)

        hours_multiplier = np.clip(game_hours / base_hours, 0.3, 3.0)
        noise = np.random.lognormal(0, 0.2, len(candidate_games))
        prediction_data['hours'] = np.maximum(0.1, base_hours * hours_multiplier * noise)

    if 'helpful' in model_features:
        base_helpful = user_profile.get('user_helpful_rate', 0.1) * 10
        prediction_data['helpful'] = np.maximum(0, np.random.poisson(base_helpful, len(candidate_games)))

    if 'funny' in model_features:
        base_funny = user_profile.get('user_funny_rate', 0.05) * 10
        prediction_data['funny'] = np.maximum(0, np.random.poisson(base_funny, len(candidate_games)))

    interaction_features = [
        ('hours_vs_user_avg', 'hours', 'user_avg_hours'),
        ('hours_vs_game_avg', 'hours', 'game_avg_hours'),
        ('hours_vs_game_median', 'hours', 'game_median_hours'),
        ('price_vs_user_avg', 'price_final', 'user_avg_price_paid')
    ]

    for feat_name, numerator, denominator in interaction_features:
        if feat_name in model_features and numerator in prediction_data.columns and denominator in prediction_data.columns:
            prediction_data[feat_name] = prediction_data[numerator] / (prediction_data[denominator] + 1e-6)

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

    except Exception as e:
        print(f"ML prediction error: {e}")
        return np.random.uniform(0.3, 0.7, len(candidate_games))


def _calculate_content_similarity(liked_games_df, candidate_games):
    feature_cols = ['positive_ratio', 'price_final']
    possible_features = [
        'game_avg_hours', 'game_median_hours', 'game_rec_rate', 'game_engagement_score',
        'win', 'mac', 'linux', 'game_positive_ratio', 'game_quality_score',
        'is_popular_game', 'is_niche_game', 'game_hours_consistency'
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

    try:
        scaler = StandardScaler()
        all_features = pd.concat([liked_features, candidate_features])
        all_features_scaled = scaler.fit_transform(all_features)

        liked_scaled = all_features_scaled[:len(liked_features)]
        candidate_scaled = all_features_scaled[len(liked_features):]
        user_profile = liked_scaled.mean(axis=0).reshape(1, -1)

        similarities = cosine_similarity(user_profile, candidate_scaled)[0]
        similarities = (similarities + 1) / 2

        return similarities
    except Exception:
        return np.random.uniform(0.3, 0.7, len(candidate_games))


def _calculate_quality_scores(candidate_games):
    scores = np.zeros(len(candidate_games))

    if 'positive_ratio' in candidate_games.columns:
        scores += candidate_games['positive_ratio'].fillna(70) / 100 * 0.4

    if 'user_reviews' in candidate_games.columns:
        review_scores = np.log1p(candidate_games['user_reviews'].fillna(1)) / 12
        review_scores = np.clip(review_scores, 0, 1)
        scores += review_scores * 0.3

    if 'game_rec_rate' in candidate_games.columns:
        scores += candidate_games['game_rec_rate'].fillna(0.6) * 0.3

    return np.clip(scores, 0, 1)


def _calculate_popularity_penalty(candidate_games, diversity_factor):
    if 'user_reviews' in candidate_games.columns and diversity_factor > 0:
        review_counts = candidate_games['user_reviews'].fillna(1)
        popularity_scores = np.log1p(review_counts) / np.log1p(review_counts.max())
        penalty = (1 - popularity_scores) * diversity_factor
        return penalty
    return np.zeros(len(candidate_games))


def get_recommendations(model_path: str, data_path: str, liked_games: list, num_recommendations: int = 10):
    try:
        if len(liked_games) < 2:
            print("Warning: Need at least 2 liked games for better recommendations")
            return []

        enriched_data = pd.read_parquet(data_path)

        recommendations = generate_game_recommendations(
            model_path=model_path,
            all_features=enriched_data,
            user_liked_games=liked_games,
            top_n=num_recommendations,
            min_reviews=50,
            diversity_factor=0.1
        )

        if recommendations.empty:
            return []

        return recommendations.to_dict('records')

    except Exception as e:
        print(f"Recommendation error: {e}")
        return []


if __name__ == "__main__":
    model_path = "data/06_models/autogluon_recommender"
    data_path = "data/04_feature/enriched_features.parquet"
    liked_games = [730, 7940]

    recommendations = get_recommendations(model_path, data_path, liked_games, 5)

    for rec in recommendations:
        print(f"{rec['rank']}. {rec['title']} (Score: {rec['recommendation_score']:.3f}, ML: {rec['ml_score']:.3f})")