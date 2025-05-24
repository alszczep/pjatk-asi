import pandas as pd
import numpy as np

def preprocess_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Improved preprocessing with better filtering"""
    recommendations['date'] = pd.to_datetime(recommendations['date'])
    recommendations['is_recommended'] = recommendations['is_recommended'].astype(bool)

    filtered = recommendations.groupby('user_id').filter(lambda x: len(x) >= 5)
    filtered = filtered.groupby('app_id').filter(lambda x: len(x) >= 10)

    print(
        f"After filtering: {len(filtered)} reviews, {filtered['user_id'].nunique()} users, {filtered['app_id'].nunique()} games")
    return filtered

def create_user_game_features(recommendations: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    print("Starting feature creation...")

    merged = recommendations.merge(games, on='app_id', how='left')
    print(f"Merged data shape: {merged.shape}")

    print("Creating user features...")
    user_stats = merged.groupby('user_id').agg({
        'app_id': 'count',  # user_total_games
        'hours': ['mean', 'std'],
        'is_recommended': 'mean',
        'helpful': 'mean',
        'funny': 'mean',
        'price_final': 'mean'
    }).reset_index()

    user_stats.columns = ['user_id', 'user_total_games', 'user_avg_hours', 'user_hours_std',
                          'user_rec_rate', 'user_helpful_rate', 'user_funny_rate', 'user_avg_price_paid']

    user_free_games = merged.groupby('user_id')['price_final'].apply(lambda x: (x == 0).mean()).reset_index()
    user_free_games.columns = ['user_id', 'user_prefers_free']

    platform_cols = ['win', 'mac', 'linux']
    if all(col in merged.columns for col in platform_cols):
        user_platform = merged.groupby('user_id')[platform_cols].sum().sum(axis=1).reset_index()
        user_platform.columns = ['user_id', 'user_platform_total']
        user_platform['user_platform_diversity'] = user_platform['user_platform_total'] / user_platform[
            'user_platform_total'].max()
    else:
        user_platform = pd.DataFrame({'user_id': user_stats['user_id'], 'user_platform_diversity': 1.0})

    user_stats = user_stats.merge(user_free_games, on='user_id', how='left')
    user_stats = user_stats.merge(user_platform[['user_id', 'user_platform_diversity']], on='user_id', how='left')

    user_stats['user_selectivity'] = 1 - user_stats['user_rec_rate']

    print("Creating game features...")
    game_stats = merged.groupby('app_id').agg({
        'hours': ['mean', 'std', 'count'],
        'is_recommended': ['mean', 'count'],
        'helpful': 'mean',
        'funny': 'mean',
        'price_final': 'first',
        'positive_ratio': 'first',
        'user_reviews': 'first'
    }).reset_index()

    game_stats.columns = ['app_id', 'game_avg_hours', 'game_hours_std', 'game_review_count',
                          'game_rec_rate', 'game_total_reviews', 'game_helpful_avg',
                          'game_funny_avg', 'game_price', 'game_positive_ratio', 'game_user_reviews']

    print("Merging features...")
    enriched = merged.merge(user_stats, on='user_id', how='left')
    enriched = enriched.merge(game_stats, on='app_id', how='left')

    print("Creating interaction features...")
    enriched['hours_vs_user_avg'] = enriched['hours'] / (enriched['user_avg_hours'] + 1e-6)
    enriched['hours_vs_game_avg'] = enriched['hours'] / (enriched['game_avg_hours'] + 1e-6)
    enriched['price_vs_user_avg'] = enriched['price_final'] / (enriched['user_avg_price_paid'] + 1e-6)

    enriched['game_quality_score'] = (enriched['game_positive_ratio'] *
                                      np.log1p(enriched['game_user_reviews']) / 100)
    enriched['is_popular_game'] = (enriched['game_user_reviews'] >
                                   enriched['game_user_reviews'].quantile(0.8)).astype(int)

    print("Filling missing values...")
    numeric_cols = enriched.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        enriched[col] = enriched[col].fillna(enriched[col].median())

    print(f"Final enriched data shape: {enriched.shape}")
    return enriched


def prepare_training_data(enriched_data: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        # User behavior features
        'user_total_games', 'user_avg_hours', 'user_hours_std', 'user_rec_rate',
        'user_helpful_rate', 'user_funny_rate', 'user_avg_price_paid',
        'user_prefers_free', 'user_platform_diversity', 'user_selectivity',

        # Game features
        'game_avg_hours', 'game_hours_std', 'game_rec_rate', 'game_helpful_avg',
        'game_funny_avg', 'game_price', 'game_positive_ratio', 'game_quality_score',
        'is_popular_game',

        # Interaction features
        'hours', 'helpful', 'funny', 'hours_vs_user_avg', 'hours_vs_game_avg',
        'price_vs_user_avg',

        # Game metadata
        'positive_ratio', 'user_reviews', 'price_final', 'discount',
        'win', 'mac', 'linux',

        # Target
        'is_recommended'
    ]

    available_cols = [col for col in feature_cols if col in enriched_data.columns]
    training_data = enriched_data[available_cols].copy()

    training_data = training_data.dropna(subset=['is_recommended'])

    print("Balancing dataset...")
    pos_mask = training_data['is_recommended'] == True
    positive_samples = training_data[pos_mask]
    negative_samples = training_data[~pos_mask]

    min_samples = min(len(positive_samples), len(negative_samples))
    max_samples = min(min_samples, 15000)  # Cap at 15k per class

    pos_sample = positive_samples.sample(n=max_samples, random_state=42)
    neg_sample = negative_samples.sample(n=max_samples, random_state=42)

    balanced_data = pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1, random_state=42)

    print(f"Balanced training data: {len(balanced_data)} samples")
    print(f"Positive ratio: {balanced_data['is_recommended'].mean():.3f}")

    return balanced_data

def create_user_game_features_chunked(recommendations: pd.DataFrame, games: pd.DataFrame,
                                      chunk_size: int = 1000000) -> pd.DataFrame:
    print(f"Processing in chunks of {chunk_size:,} rows...")

    merged = recommendations.merge(games, on='app_id', how='left')

    print("Creating user features (full dataset)...")
    user_stats = merged.groupby('user_id').agg({
        'app_id': 'count',
        'hours': ['mean', 'std'],
        'is_recommended': 'mean',
        'helpful': 'mean',
        'funny': 'mean',
        'price_final': 'mean'
    }).reset_index()

    user_stats.columns = ['user_id', 'user_total_games', 'user_avg_hours', 'user_hours_std',
                          'user_rec_rate', 'user_helpful_rate', 'user_funny_rate', 'user_avg_price_paid']

    user_free_games = merged.groupby('user_id')['price_final'].apply(lambda x: (x == 0).mean()).reset_index()
    user_free_games.columns = ['user_id', 'user_prefers_free']
    user_stats = user_stats.merge(user_free_games, on='user_id', how='left')
    user_stats['user_selectivity'] = 1 - user_stats['user_rec_rate']

    print("Creating game features (full dataset)...")
    game_stats = merged.groupby('app_id').agg({
        'hours': ['mean', 'std', 'count'],
        'is_recommended': ['mean', 'count'],
        'helpful': 'mean',
        'funny': 'mean',
        'price_final': 'first',
        'positive_ratio': 'first',
        'user_reviews': 'first'
    }).reset_index()

    game_stats.columns = ['app_id', 'game_avg_hours', 'game_hours_std', 'game_review_count',
                          'game_rec_rate', 'game_total_reviews', 'game_helpful_avg',
                          'game_funny_avg', 'game_price', 'game_positive_ratio', 'game_user_reviews']

    chunks = []
    for i in range(0, len(merged), chunk_size):
        print(f"Processing chunk {i // chunk_size + 1}/{(len(merged) - 1) // chunk_size + 1}")
        chunk = merged.iloc[i:i + chunk_size].copy()

        chunk = chunk.merge(user_stats, on='user_id', how='left')
        chunk = chunk.merge(game_stats, on='app_id', how='left')

        chunk['hours_vs_user_avg'] = chunk['hours'] / (chunk['user_avg_hours'] + 1e-6)
        chunk['hours_vs_game_avg'] = chunk['hours'] / (chunk['game_avg_hours'] + 1e-6)
        chunk['price_vs_user_avg'] = chunk['price_final'] / (chunk['user_avg_price_paid'] + 1e-6)

        chunks.append(chunk)

    print("Combining chunks...")
    enriched = pd.concat(chunks, ignore_index=True)

    enriched['game_quality_score'] = (enriched['game_positive_ratio'] *
                                      np.log1p(enriched['game_user_reviews']) / 100)
    enriched['is_popular_game'] = (enriched['game_user_reviews'] >
                                   enriched['game_user_reviews'].quantile(0.8)).astype(int)

    numeric_cols = enriched.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        enriched[col] = enriched[col].fillna(enriched[col].median())

    return enriched