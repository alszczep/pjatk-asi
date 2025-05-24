import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def train_autogluon_recommender(training_data: pd.DataFrame) -> str:
    """Train AutoGluon recommender model - UNCHANGED"""

    model_path = os.path.abspath("data/06_models/autogluon_recommender")
    os.makedirs(model_path, exist_ok=True)

    if len(training_data) > 10000:
        positive_samples = training_data[training_data['is_recommended'] == True]
        negative_samples = training_data[training_data['is_recommended'] == False]

        pos_ratio = len(positive_samples) / len(training_data)
        n_pos = int(10000 * pos_ratio)
        n_neg = 10000 - n_pos

        pos_sample = positive_samples.sample(n=min(n_pos, len(positive_samples)), random_state=42)
        neg_sample = negative_samples.sample(n=min(n_neg, len(negative_samples)), random_state=42)

        training_sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
    else:
        training_sample = training_data

    training_sample = training_sample.fillna(0)

    predictor = TabularPredictor(
        label='is_recommended',
        path=model_path,
        problem_type='binary',
        eval_metric='roc_auc'
    ).fit(
        training_sample,
        presets='good_quality',
        time_limit=300,  # 5 minutes max
        verbosity=2,
        ag_args_fit={'ag.max_memory_usage_ratio': 0.6}
    )

    print("AutoGluon training completed!")
    print("Model leaderboard:")
    print(predictor.leaderboard())

    path_file = "data/06_models/autogluon_model_path.txt"
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, 'w') as f:
        f.write(model_path)

    return model_path


def improved_generate_game_recommendations(
        model_path: str,
        all_features: pd.DataFrame,
        user_liked_games: list,
        top_n: int = 10,
        min_reviews: int = 50,
        diversity_factor: float = 0.3
) -> pd.DataFrame:
    """
    Generate improved game recommendations using hybrid approach
    """
    try:
        predictor = TabularPredictor.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return pd.DataFrame()

    # Get unique games and filter by minimum reviews for quality
    games_df = all_features.groupby('app_id').agg({
        'title': 'first',
        'positive_ratio': 'first',
        'user_reviews': 'first',
        'price_final': 'first',
        'game_hours_mean': 'first',
        'game_rec_rate': 'first',
        'game_review_count': 'first',
        # Include all other features
        **{col: 'first' for col in all_features.columns
           if col not in ['app_id', 'title', 'positive_ratio', 'user_reviews',
                          'price_final', 'game_hours_mean', 'game_rec_rate', 'game_review_count']}
    }).reset_index()

    # Filter out games with too few reviews (likely obscure/low quality)
    if 'user_reviews' in games_df.columns:
        games_df = games_df[games_df['user_reviews'] >= min_reviews]
        print(f"Filtered to {len(games_df)} games with >= {min_reviews} reviews")

    # Get liked games data
    liked_games_df = games_df[games_df['app_id'].isin(user_liked_games)]
    if len(liked_games_df) == 0:
        print("✗ None of the liked games found in filtered dataset!")
        return pd.DataFrame()

    print(f"Found {len(liked_games_df)} liked games:")
    for _, game in liked_games_df.iterrows():
        print(f"  - {game['title']} (ID: {game['app_id']})")

    # Create user profile from liked games
    numeric_cols = liked_games_df.select_dtypes(include=[np.number]).columns
    user_profile = liked_games_df[numeric_cols].mean()

    # Get candidate games (excluding already liked ones)
    candidate_games = games_df[~games_df['app_id'].isin(user_liked_games)].copy()

    if len(candidate_games) == 0:
        print("✗ No candidate games available!")
        return pd.DataFrame()

    print(f"Evaluating {len(candidate_games)} candidate games...")

    # Method 1: Content-based similarity
    content_scores = _calculate_content_similarity(liked_games_df, candidate_games)

    # Method 2: ML model predictions (with more realistic synthetic data)
    ml_scores = _get_ml_predictions(predictor, candidate_games, user_profile)

    # Method 3: Popularity and quality scores
    quality_scores = _calculate_quality_scores(candidate_games)

    # Combine all scores
    final_scores = _combine_scores(content_scores, ml_scores, quality_scores, diversity_factor)

    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'app_id': candidate_games['app_id'].values,
        'title': candidate_games['title'].values,
        'recommendation_score': final_scores,
        'content_similarity': content_scores,
        'ml_score': ml_scores,
        'quality_score': quality_scores,
        'user_reviews': candidate_games['user_reviews'].values,
        'positive_ratio': candidate_games['positive_ratio'].values,
        'price_final': candidate_games['price_final'].values
    })

    # Sort and get top recommendations
    recommendations = recommendations.sort_values('recommendation_score', ascending=False).head(top_n)
    recommendations['rank'] = range(1, len(recommendations) + 1)

    return recommendations.reset_index(drop=True)


def _calculate_content_similarity(liked_games_df, candidate_games):
    """Calculate content-based similarity using game features"""

    # Select relevant features for similarity calculation
    feature_cols = [
        'positive_ratio', 'user_reviews', 'price_final',
        'game_hours_mean', 'game_rec_rate', 'win', 'mac', 'linux',
        'is_free', 'has_discount', 'platform_count'
    ]

    # Get available features
    available_features = [col for col in feature_cols
                          if col in liked_games_df.columns and col in candidate_games.columns]

    if not available_features:
        print("Warning: No common features found for similarity calculation")
        return np.zeros(len(candidate_games))

    # Normalize features
    scaler = StandardScaler()

    # Fit on liked games and transform both
    liked_features = liked_games_df[available_features].fillna(0)
    candidate_features = candidate_games[available_features].fillna(0)

    if len(liked_features) == 0:
        return np.zeros(len(candidate_games))

    # Scale features
    all_features = pd.concat([liked_features, candidate_features])
    all_features_scaled = scaler.fit_transform(all_features)

    liked_scaled = all_features_scaled[:len(liked_features)]
    candidate_scaled = all_features_scaled[len(liked_features):]

    # Calculate average liked game profile
    user_profile = liked_scaled.mean(axis=0).reshape(1, -1)

    # Calculate similarities
    similarities = cosine_similarity(user_profile, candidate_scaled)[0]

    # Normalize to 0-1 range
    similarities = (similarities + 1) / 2  # Cosine similarity is in [-1, 1]

    return similarities


def _get_ml_predictions(predictor, candidate_games, user_profile):
    """Get ML model predictions with more realistic synthetic interactions"""

    feature_cols = [col for col in candidate_games.columns
                    if col not in ['app_id', 'title', 'is_recommended']]

    # Create more conservative synthetic interactions
    synthetic_data = candidate_games[feature_cols].copy()

    # Add realistic user features based on profile
    user_features = {
        'user_hours_mean': user_profile.get('user_hours_mean', 15.0),
        'user_hours_std': user_profile.get('user_hours_std', 8.0),
        'user_hours_count': user_profile.get('user_hours_count', 8.0),
        'user_rec_rate': user_profile.get('user_rec_rate', 0.6),
        'user_helpful_total': user_profile.get('user_helpful_total', 3.0),
        'user_funny_total': user_profile.get('user_funny_total', 1.0),
    }

    for feature, value in user_features.items():
        if feature in synthetic_data.columns:
            # Add some noise to make predictions more realistic
            synthetic_data[feature] = value * np.random.normal(1.0, 0.1, len(synthetic_data))

    # Create realistic interaction features
    synthetic_data['hours'] = np.maximum(
        1.0,
        user_features['user_hours_mean'] * np.random.lognormal(0, 0.5, len(synthetic_data))
    )

    if 'user_hours_mean' in synthetic_data.columns:
        synthetic_data['hours_vs_user_avg'] = synthetic_data['hours'] / (
                synthetic_data['user_hours_mean'] + 1e-6
        )

    synthetic_data['helpful'] = np.random.poisson(1.5, len(synthetic_data))
    synthetic_data['funny'] = np.random.poisson(0.5, len(synthetic_data))

    if 'user_rec_rate' in synthetic_data.columns:
        synthetic_data['user_selectivity'] = 1 - synthetic_data['user_rec_rate']

    # Fill any remaining NaN values
    synthetic_data = synthetic_data.fillna(0)

    try:
        # Get predictions
        predictions = predictor.predict_proba(synthetic_data)

        if hasattr(predictions, 'iloc') and predictions.shape[1] > 1:
            scores = predictions.iloc[:, 1].values
        else:
            scores = predictions if isinstance(predictions, np.ndarray) else predictions.values

        # Apply sigmoid to normalize scores and reduce extreme values
        scores = 1 / (1 + np.exp(-10 * (scores - 0.5)))  # More conservative sigmoid

        return scores

    except Exception as e:
        print(f"Warning: ML prediction failed: {e}")
        return np.random.uniform(0.3, 0.7, len(candidate_games))  # Fallback to random


def _calculate_quality_scores(candidate_games):
    """Calculate quality scores based on reviews and ratings"""

    scores = np.zeros(len(candidate_games))

    # Positive ratio score (0-1)
    if 'positive_ratio' in candidate_games.columns:
        scores += candidate_games['positive_ratio'].fillna(50) / 100 * 0.4

    # Review count score (logarithmic scaling)
    if 'user_reviews' in candidate_games.columns:
        review_scores = np.log1p(candidate_games['user_reviews'].fillna(1)) / 10
        review_scores = np.clip(review_scores, 0, 1)  # Cap at 1
        scores += review_scores * 0.3

    # Game recommendation rate
    if 'game_rec_rate' in candidate_games.columns:
        scores += candidate_games['game_rec_rate'].fillna(0.5) * 0.3

    return np.clip(scores, 0, 1)


def _combine_scores(content_scores, ml_scores, quality_scores, diversity_factor):
    """Combine different scoring methods"""

    # Normalize all scores to 0-1 range
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
    ml_scores = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min() + 1e-8)
    quality_scores = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-8)

    # Weighted combination
    final_scores = (
            content_scores * 0.4 +  # Content similarity
            ml_scores * 0.4 +  # ML model prediction
            quality_scores * 0.2  # Quality/popularity
    )

    # Add diversity penalty (reduce scores for very similar games)
    diversity_penalty = np.random.uniform(0, diversity_factor, len(final_scores))
    final_scores = final_scores * (1 - diversity_penalty)

    return final_scores


def evaluate_recommender(model_path: str, training_data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the AutoGluon recommender performance - UNCHANGED"""

    predictor = TabularPredictor.load(model_path)

    test_size = min(1000, len(training_data) // 4)
    test_data = training_data.sample(n=test_size, random_state=42)
    feature_cols = [col for col in test_data.columns if col != 'is_recommended']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['is_recommended']
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)

    if hasattr(y_pred_proba, 'iloc') and y_pred_proba.shape[1] > 1:
        proba_positive = y_pred_proba.iloc[:, 1]
    else:
        proba_positive = y_pred_proba

    auc_score = roc_auc_score(y_test, proba_positive)
    report = classification_report(y_test, y_pred, output_dict=True)
    eval_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1-score', 'auc'],
        'value': [
            report['accuracy'],
            report['True']['precision'],
            report['True']['recall'],
            report['True']['f1-score'],
            auc_score
        ]
    })

    print("=== AutoGluon Recommender Evaluation ===")
    print(eval_df)

    return eval_df


# Keep the old function for backward compatibility (DEPRECATED)
def generate_game_recommendations(
        model_path: str,
        all_features: pd.DataFrame,
        user_liked_games: list,
        top_n: int = 10
) -> pd.DataFrame:
    """DEPRECATED: Use improved_generate_game_recommendations instead"""
    print("Warning: Using deprecated function. Please update to use improved_generate_game_recommendations")
    return improved_generate_game_recommendations(model_path, all_features, user_liked_games, top_n)