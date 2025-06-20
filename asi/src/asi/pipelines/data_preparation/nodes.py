import pandas as pd
import numpy as np
import gc


def preprocess_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
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
        'app_id': 'count',
        'hours': ['mean', 'std', 'max', 'min'],
        'is_recommended': 'mean',
        'helpful': 'mean',
        'funny': 'mean',
        'price_final': ['mean', 'std']
    }).reset_index()

    user_stats.columns = ['user_id', 'user_total_games', 'user_avg_hours', 'user_hours_std',
                          'user_max_hours', 'user_min_hours', 'user_rec_rate',
                          'user_helpful_rate', 'user_funny_rate', 'user_avg_price_paid', 'user_price_std']

    user_free_games = merged.groupby('user_id')['price_final'].apply(lambda x: (x == 0).mean()).reset_index()
    user_free_games.columns = ['user_id', 'user_prefers_free']

    user_genre_diversity = merged.groupby('user_id').apply(
        lambda x: len(x['app_id'].unique()) / len(x) if len(x) > 0 else 0
    ).reset_index()
    user_genre_diversity.columns = ['user_id', 'user_game_diversity']

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
    user_stats = user_stats.merge(user_genre_diversity, on='user_id', how='left')

    user_stats['user_selectivity'] = 1 - user_stats['user_rec_rate']
    user_stats['user_hours_coefficient_variation'] = user_stats['user_hours_std'] / (
            user_stats['user_avg_hours'] + 1e-6)
    user_stats['user_price_sensitivity'] = user_stats['user_price_std'] / (user_stats['user_avg_price_paid'] + 1e-6)

    print("Creating game features...")
    game_stats = merged.groupby('app_id').agg({
        'hours': ['mean', 'std', 'count', 'median'],
        'is_recommended': ['mean', 'count'],
        'helpful': ['mean', 'std'],
        'funny': ['mean', 'std'],
        'price_final': 'first',
        'positive_ratio': 'first',
        'user_reviews': 'first'
    }).reset_index()

    game_stats.columns = ['app_id', 'game_avg_hours', 'game_hours_std', 'game_review_count',
                          'game_median_hours', 'game_rec_rate', 'game_total_reviews',
                          'game_helpful_avg', 'game_helpful_std', 'game_funny_avg', 'game_funny_std',
                          'game_price', 'game_positive_ratio', 'game_user_reviews']

    game_stats['game_engagement_score'] = (game_stats['game_avg_hours'] * game_stats['game_rec_rate'] *
                                           np.log1p(game_stats['game_total_reviews'])) / 100
    game_stats['game_hours_consistency'] = 1 / (
            1 + game_stats['game_hours_std'] / (game_stats['game_avg_hours'] + 1e-6))

    print("Merging features...")
    enriched = merged.merge(user_stats, on='user_id', how='left')
    enriched = enriched.merge(game_stats, on='app_id', how='left')

    print("Creating interaction features...")
    enriched['hours_vs_user_avg'] = enriched['hours'] / (enriched['user_avg_hours'] + 1e-6)
    enriched['hours_vs_game_avg'] = enriched['hours'] / (enriched['game_avg_hours'] + 1e-6)
    enriched['hours_vs_game_median'] = enriched['hours'] / (enriched['game_median_hours'] + 1e-6)
    enriched['price_vs_user_avg'] = enriched['price_final'] / (enriched['user_avg_price_paid'] + 1e-6)

    enriched['user_game_price_match'] = np.abs(enriched['price_final'] - enriched['user_avg_price_paid']) / (
            enriched['user_avg_price_paid'] + 1e-6)
    enriched['user_game_quality_match'] = np.abs(enriched['game_positive_ratio'] - enriched['user_rec_rate'] * 100)

    enriched['game_quality_score'] = (enriched['game_positive_ratio'] *
                                      np.log1p(enriched['game_user_reviews']) / 100)
    enriched['is_popular_game'] = (enriched['game_user_reviews'] >
                                   enriched['game_user_reviews'].quantile(0.8)).astype(int)
    enriched['is_niche_game'] = (enriched['game_user_reviews'] <
                                 enriched['game_user_reviews'].quantile(0.2)).astype(int)

    enriched['playtime_category'] = pd.cut(enriched['hours'],
                                           bins=[0, 1, 5, 20, 100, float('inf')],
                                           labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    enriched['playtime_category'] = enriched['playtime_category'].astype(str)

    print("Filling missing values...")
    numeric_cols = enriched.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        enriched[col] = enriched[col].fillna(enriched[col].median())

    categorical_cols = enriched.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'playtime_category':
            enriched[col] = enriched[col].fillna('unknown')

    print(f"Final enriched data shape: {enriched.shape}")
    return enriched


def prepare_training_data(enriched_data: pd.DataFrame, max_samples: int = 50000) -> pd.DataFrame:
    print(f"Starting with {len(enriched_data)} total samples")

    essential_features = [

        'user_total_games', 'user_avg_hours', 'user_rec_rate', 'user_helpful_rate',
        'user_avg_price_paid', 'user_prefers_free', 'user_selectivity',

        'game_avg_hours', 'game_rec_rate', 'game_price', 'game_positive_ratio',
        'game_engagement_score', 'is_popular_game', 'is_niche_game',

        'hours', 'helpful', 'funny', 'hours_vs_user_avg', 'hours_vs_game_avg',
        'price_vs_user_avg', 'user_game_quality_match',

        'positive_ratio', 'user_reviews', 'price_final', 'discount',
        'win', 'mac', 'linux',

        'is_recommended'
    ]

    available_cols = [col for col in essential_features if col in enriched_data.columns]
    print(f"Using {len(available_cols)} features out of {len(essential_features)} requested")

    working_data = enriched_data[available_cols].copy()

    working_data = working_data.dropna(subset=['is_recommended'])
    print(f"After dropping missing targets: {len(working_data)} samples")

    if len(working_data) > max_samples * 3:
        print(f"Data too large ({len(working_data)}), sampling {max_samples * 3} rows first")
        working_data = working_data.sample(n=max_samples * 3, random_state=42)
        gc.collect()

    if 'playtime_category' in enriched_data.columns:
        playtime_cats = enriched_data['playtime_category'].fillna('unknown')

        playtime_dummies = pd.get_dummies(playtime_cats, prefix='playtime', sparse=True)

        top_categories = playtime_dummies.sum().nlargest(5).index
        for col in top_categories:
            working_data[col.replace('playtime_', 'playtime_')] = playtime_dummies[col].astype('int8')

    print("Balancing dataset with memory-efficient approach...")

    positive_mask = working_data['is_recommended'] == True
    positive_count = positive_mask.sum()
    negative_count = len(working_data) - positive_count

    print(f"Original distribution - Positive: {positive_count}, Negative: {negative_count}")

    min_class_size = min(positive_count, negative_count)
    target_per_class = min(min_class_size, max_samples // 2)

    print(f"Target samples per class: {target_per_class}")

    positive_indices = working_data[positive_mask].index
    negative_indices = working_data[~positive_mask].index

    np.random.seed(42)
    selected_positive = np.random.choice(positive_indices, size=target_per_class, replace=False)
    selected_negative = np.random.choice(negative_indices, size=target_per_class, replace=False)

    selected_indices = np.concatenate([selected_positive, selected_negative])
    balanced_data = working_data.loc[selected_indices].copy()

    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    del working_data
    gc.collect()

    print(f"Final balanced dataset: {len(balanced_data)} samples")
    print(f"Positive ratio: {balanced_data['is_recommended'].mean():.3f}")
    print(f"Memory usage: {balanced_data.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

    return balanced_data
