import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import warnings
from collections import Counter

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
        model_features = [col for col in predictor.feature_metadata.get_features()
                          if col != 'is_recommended']
        print(f"Model expects {len(model_features)} features")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return pd.DataFrame()

    user_data = all_features[all_features['app_id'].isin(user_liked_games)]
    if len(user_data) == 0:
        print("No data found for liked games")
        return pd.DataFrame()


    user_profile = _create_robust_user_profile(user_data, all_features)

    candidate_games = _get_candidate_games(all_features, user_liked_games, min_reviews)
    if len(candidate_games) == 0:
        print("No candidate games found")
        return pd.DataFrame()

    prediction_data = _create_prediction_features(candidate_games, user_profile, model_features, all_features)

    try:
        predictions = predictor.predict_proba(prediction_data)

        if hasattr(predictions, 'iloc') and predictions.shape[1] > 1:
            scores = predictions.iloc[:, 1].values
        else:
            scores = predictions if isinstance(predictions, np.ndarray) else predictions.values

        scores = np.clip(scores, 0, 1)
        print(f"ML scores - Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}")

        recommendations = pd.DataFrame({
            'app_id': candidate_games['app_id'].values,
            'title': candidate_games['title'].values,
            'recommendation_score': scores,
            'ml_score': scores,
            'user_reviews': candidate_games.get('user_reviews', 0),
            'positive_ratio': candidate_games.get('positive_ratio', 0),
            'price_final': candidate_games.get('price_final', 0)
        })

        recommendations = _apply_improved_content_boosting(
            recommendations, user_profile, user_data
        )

        recommendations = recommendations.sort_values('recommendation_score', ascending=False).head(top_n)
        recommendations['rank'] = range(1, len(recommendations) + 1)

        print(f"Generated {len(recommendations)} recommendations")
        print(
            f"Final score range: {recommendations['recommendation_score'].min():.3f} - {recommendations['recommendation_score'].max():.3f}")

        return recommendations.reset_index(drop=True)

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def _create_robust_user_profile(user_data, all_features):
    profile = {}

    profile['user_total_games'] = len(user_data)
    profile['user_avg_hours'] = user_data['hours'].mean()
    profile['user_hours_std'] = user_data['hours'].std() if len(user_data) > 1 else 0.0
    profile['user_max_hours'] = user_data['hours'].max()
    profile['user_min_hours'] = user_data['hours'].min()
    profile['user_rec_rate'] = user_data['is_recommended'].mean()
    profile['user_helpful_rate'] = user_data['helpful'].mean()
    profile['user_funny_rate'] = user_data['funny'].mean()
    profile['user_avg_price_paid'] = user_data['price_final'].mean()
    profile['user_price_std'] = user_data['price_final'].std() if len(user_data) > 1 else 0.0
    profile['user_prefers_free'] = (user_data['price_final'] == 0).mean()

    profile['user_selectivity'] = 1 - profile['user_rec_rate']
    profile['user_hours_coefficient_variation'] = profile['user_hours_std'] / (profile['user_avg_hours'] + 1e-6)
    profile['user_price_sensitivity'] = profile['user_price_std'] / (profile['user_avg_price_paid'] + 1e-6)
    profile['user_game_diversity'] = len(user_data['app_id'].unique()) / len(user_data)

    platform_cols = ['win', 'mac', 'linux']
    for col in platform_cols:
        if col in user_data.columns:
            profile[f'user_{col}_preference'] = user_data[col].mean()
        else:
            profile[f'user_{col}_preference'] = 0.0

    if 'positive_ratio' in user_data.columns:
        profile['user_quality_preference'] = user_data['positive_ratio'].mean()

    for key, value in profile.items():
        if pd.isna(value):
            profile[key] = 0.0

    return profile

def _get_candidate_games(all_features, user_liked_games, min_reviews):
    agg_dict = {
        'title': 'first',
        'positive_ratio': 'first',
        'user_reviews': 'first',
        'price_final': 'first',
        'win': 'first',
        'mac': 'first',
        'linux': 'first',
        'discount': 'first'
    }

    game_features = [col for col in all_features.columns if col.startswith('game_')]
    for col in game_features:
        agg_dict[col] = 'first'

    other_features = ['hours', 'helpful', 'funny']
    for col in other_features:
        if col in all_features.columns:
            agg_dict[col] = 'mean'

    games_df = all_features.groupby('app_id').agg(agg_dict).reset_index()

    games_df = games_df[
        (games_df['user_reviews'] >= min_reviews) &
        (~games_df['app_id'].isin(user_liked_games))
        ]

    print(f"Candidate games after filtering: {len(games_df)}")
    return games_df


def _create_prediction_features(candidate_games, user_profile, model_features, all_features):
    prediction_data = candidate_games.copy()

    for feature, value in user_profile.items():
        if feature in model_features:
            prediction_data[feature] = value

    if 'hours' in model_features:

        base_hours = user_profile.get('user_avg_hours', 10.0)

        if 'hours' in prediction_data.columns:

            prediction_data['hours'] = 0.7 * base_hours + 0.3 * prediction_data['hours']
        else:

            if 'positive_ratio' in prediction_data.columns:
                quality_multiplier = np.clip(prediction_data['positive_ratio'] / 75.0, 0.5, 1.5)
                prediction_data['hours'] = base_hours * quality_multiplier
            else:
                prediction_data['hours'] = base_hours

    if 'hours_vs_user_avg' in model_features:
        prediction_data['hours_vs_user_avg'] = prediction_data.get('hours',
                                                                   user_profile.get('user_avg_hours', 10.0)) / (
                                                       user_profile.get('user_avg_hours', 10.0) + 1e-6)

    if 'hours_vs_game_avg' in model_features:
        if 'game_avg_hours' in prediction_data.columns:
            prediction_data['hours_vs_game_avg'] = prediction_data.get('hours', 10.0) / (
                    prediction_data['game_avg_hours'] + 1e-6)
        else:
            prediction_data['hours_vs_game_avg'] = 1.0

    if 'hours_vs_game_median' in model_features:
        if 'game_median_hours' in prediction_data.columns:
            prediction_data['hours_vs_game_median'] = prediction_data.get('hours', 10.0) / (
                    prediction_data['game_median_hours'] + 1e-6)
        else:
            prediction_data['hours_vs_game_median'] = 1.0

    if 'price_vs_user_avg' in model_features:
        prediction_data['price_vs_user_avg'] = prediction_data['price_final'] / (
                user_profile.get('user_avg_price_paid', 10.0) + 1e-6)

    if 'helpful' in model_features:
        if 'helpful' not in prediction_data.columns:
            prediction_data['helpful'] = user_profile.get('user_helpful_rate', 1.0)

    if 'funny' in model_features:
        if 'funny' not in prediction_data.columns:
            prediction_data['funny'] = user_profile.get('user_funny_rate', 0.5)

    if 'user_game_quality_match' in model_features and 'positive_ratio' in prediction_data.columns:
        user_quality_pref = user_profile.get('user_quality_preference', user_profile.get('user_rec_rate', 0.5) * 100)
        prediction_data['user_game_quality_match'] = np.abs(prediction_data['positive_ratio'] - user_quality_pref)

    if 'user_game_price_match' in model_features:
        user_avg_price = user_profile.get('user_avg_price_paid', 10.0)
        prediction_data['user_game_price_match'] = np.abs(prediction_data['price_final'] - user_avg_price) / (
                user_avg_price + 1e-6)

    if 'game_quality_score' in model_features and 'positive_ratio' in prediction_data.columns and 'user_reviews' in prediction_data.columns:
        prediction_data['game_quality_score'] = (
                prediction_data['positive_ratio'] * np.log1p(prediction_data['user_reviews']) / 100)

    if 'is_popular_game' in model_features and 'user_reviews' in prediction_data.columns:
        popularity_threshold = prediction_data['user_reviews'].quantile(0.8)
        prediction_data['is_popular_game'] = (prediction_data['user_reviews'] > popularity_threshold).astype(int)

    if 'is_niche_game' in model_features and 'user_reviews' in prediction_data.columns:
        niche_threshold = prediction_data['user_reviews'].quantile(0.2)
        prediction_data['is_niche_game'] = (prediction_data['user_reviews'] < niche_threshold).astype(int)

    playtime_features = [f for f in model_features if f.startswith('playtime_')]
    if playtime_features and 'hours' in prediction_data.columns:
        hours_values = prediction_data['hours'].values

        playtime_mapping = {
            'playtime_very_short': hours_values < 1,
            'playtime_short': (hours_values >= 1) & (hours_values < 5),
            'playtime_medium': (hours_values >= 5) & (hours_values < 20),
            'playtime_long': (hours_values >= 20) & (hours_values < 100),
            'playtime_very_long': hours_values >= 100
        }

        for feature in playtime_features:
            if feature in playtime_mapping:
                prediction_data[feature] = playtime_mapping[feature].astype(int)
            else:
                prediction_data[feature] = 0

    for feature in model_features:
        if feature not in prediction_data.columns:
            if 'user_' in feature and feature in user_profile:
                prediction_data[feature] = user_profile[feature]
            elif 'game_' in feature:
                prediction_data[feature] = 0.0
            elif feature in ['win', 'mac', 'linux']:
                prediction_data[feature] = user_profile.get(f'user_{feature}_preference',
                                                            1.0 if feature == 'win' else 0.0)
            else:
                prediction_data[feature] = 0.0

    prediction_input = prediction_data[model_features].fillna(0)
    return prediction_input


def _apply_improved_content_boosting(recommendations, user_profile, user_data):
    original_scores = recommendations['recommendation_score'].copy()

    user_liked_quality = user_data['positive_ratio'].mean() if 'positive_ratio' in user_data.columns else 75
    quality_diff = recommendations['positive_ratio'] - user_liked_quality
    quality_boost = np.tanh(quality_diff / 20) * 0.08
    recommendations['recommendation_score'] += quality_boost

    user_avg_price = user_profile.get('user_avg_price_paid', 10.0)
    price_similarity = np.exp(-np.abs(recommendations['price_final'] - user_avg_price) / (user_avg_price + 1))
    price_boost = price_similarity * 0.05
    recommendations['recommendation_score'] += price_boost

    log_reviews = np.log1p(recommendations['user_reviews'])
    max_log_reviews = log_reviews.max()
    if max_log_reviews > 0:
        popularity_boost = log_reviews / max_log_reviews * 0.04
        recommendations['recommendation_score'] += popularity_boost

    if any(f'user_{platform}_preference' in user_profile for platform in ['win', 'mac', 'linux']):
        platform_score = 0
        for platform in ['win', 'mac', 'linux']:
            user_pref = user_profile.get(f'user_{platform}_preference', 0)
            if platform in recommendations.columns:
                platform_score += recommendations[platform] * user_pref
        recommendations['recommendation_score'] += platform_score * 0.03

    user_selectivity = user_profile.get('user_selectivity', 0.5)
    if user_selectivity > 0.6:
        selectivity_boost = (recommendations['ml_score'] - 0.5) * user_selectivity * 0.05
        recommendations['recommendation_score'] += np.maximum(0, selectivity_boost)

    min_score = recommendations['recommendation_score'].min()
    max_score = recommendations['recommendation_score'].max()

    if max_score > min_score:
        recommendations['recommendation_score'] = (recommendations['recommendation_score'] - min_score) / (
                max_score - min_score)
    else:
        recommendations['recommendation_score'] = 0.5
    return recommendations


def get_recommendations(model_path: str, data_path: str, liked_games: list, num_recommendations: int = 10):
    try:
        if len(liked_games) < 1:
            print("Error: Need at least 1 liked game for recommendations")
            return []

        if len(liked_games) < 2:
            print("Warning: Having at least 2 liked games provides better recommendations")

        print(f"Loading data from {data_path}")
        enriched_data = pd.read_parquet(data_path)
        print(f"Loaded {len(enriched_data)} rows with {len(enriched_data.columns)} columns")

        available_games = set(enriched_data['app_id'].unique())
        valid_liked_games = [game for game in liked_games if game in available_games]

        if not valid_liked_games:
            print(f"Error: None of the liked games {liked_games} found in the dataset")
            return []

        if len(valid_liked_games) < len(liked_games):
            missing_games = set(liked_games) - set(valid_liked_games)
            print(f"Warning: Games {missing_games} not found in dataset, using {valid_liked_games}")

        recommendations = generate_game_recommendations(
            model_path=model_path,
            all_features=enriched_data,
            user_liked_games=valid_liked_games,
            top_n=num_recommendations,
            min_reviews=50
        )

        if recommendations.empty:
            print("No recommendations generated")
            return []

        return recommendations.to_dict('records')

    except Exception as e:
        print(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    model_path = "data/06_models/autogluon_recommender"
    data_path = "data/04_feature/enriched_features.parquet"
    liked_games = [311210, 1938090, 202970, 10180, 42700]

    recommendations = get_recommendations(model_path, data_path, liked_games, 5)

    for rec in recommendations:
        print(
            f"{rec['rank']}. {rec['title']} (Score: {rec['recommendation_score']:.3f}, ML: {rec['ml_score']:.3f})")
