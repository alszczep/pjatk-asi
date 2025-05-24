
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import the improved functions
from improved_autogluon_recommender import (
    preprocess_recommendations,
    create_improved_features,
    prepare_balanced_training_data,
    train_improved_autogluon_recommender,
    test_improved_recommendations
)


def run_complete_pipeline():
    """Run the complete improved pipeline"""

    print("🚀 Starting Improved AutoGluon Recommendation Pipeline")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n📁 Step 1: Loading raw data...")
    try:
        recommendations = pd.read_csv("data/01_raw/recommendations.csv")  # Adjust path as needed
        games = pd.read_csv("data/01_raw/games.csv")  # Adjust path as needed
        print(f"✓ Loaded {len(recommendations)} recommendations and {len(games)} games")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please ensure your data files are in the correct location")
        return

    # Step 2: Preprocess recommendations
    print("\n🔄 Step 2: Preprocessing recommendations...")
    try:
        filtered_recommendations = preprocess_recommendations(recommendations)
        print(f"✓ Filtered to {len(filtered_recommendations)} high-quality recommendations")
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return

    # Step 3: Create improved features
    print("\n⚙️ Step 3: Creating improved features...")
    try:
        enriched_features = create_improved_features(filtered_recommendations, games)
        print(f"✓ Created enriched dataset with {len(enriched_features.columns)} features")

        # Save enriched features
        os.makedirs("data/04_feature", exist_ok=True)
        enriched_features.to_parquet("data/04_feature/improved_enriched_features.parquet")
        print("✓ Saved enriched features")

    except Exception as e:
        print(f"✗ Error in feature creation: {e}")
        return

    # Step 4: Prepare balanced training data
    print("\n⚖️ Step 4: Preparing balanced training data...")
    try:
        training_data = prepare_balanced_training_data(enriched_features)
        print(f"✓ Prepared balanced training data with {len(training_data)} samples")

        # Save training data
        training_data.to_parquet("data/05_model_input/improved_training_data.parquet")
        print("✓ Saved training data")

    except Exception as e:
        print(f"✗ Error in data preparation: {e}")
        return

    # Step 5: Train improved model
    print("\n🧠 Step 5: Training improved AutoGluon model...")
    try:
        model_path = train_improved_autogluon_recommender(training_data)
        print(f"✓ Model trained and saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return

    # Step 6: Test the model
    print("\n🧪 Step 6: Testing improved recommendations...")
    try:
        test_improved_recommendations()
        print("✓ Testing completed")
    except Exception as e:
        print(f"✗ Error in testing: {e}")
        return

    print("\n🎉 Pipeline completed successfully!")
    print("\nKey improvements made:")
    print("• Better data filtering (min 5 reviews per user, 10 per game)")
    print("• Improved feature engineering without data leakage")
    print("• Balanced training data (equal positive/negative samples)")
    print("• Better AutoGluon configuration to reduce overfitting")
    print("• Improved user profiling for recommendations")
    print("• Quality filtering for recommended games")


def analyze_current_model_issues():
    """Analyze issues with the current model"""

    print("🔍 Analyzing Current Model Issues")
    print("=" * 40)

    try:
        # Load current data
        enriched_data = pd.read_parquet("data/04_feature/enriched_features.parquet")

        print(f"Current dataset size: {len(enriched_data)} rows")
        print(f"Number of features: {len(enriched_data.columns)}")
        print(f"Number of unique users: {enriched_data['user_id'].nunique()}")
        print(f"Number of unique games: {enriched_data['app_id'].nunique()}")

        # Analyze recommendation distribution
        rec_dist = enriched_data['is_recommended'].value_counts(normalize=True)
        print(f"\nRecommendation distribution:")
        print(f"• Positive: {rec_dist[True]:.3f}")
        print(f"• Negative: {rec_dist[False]:.3f}")

        # Analyze game popularity
        game_counts = enriched_data['app_id'].value_counts()
        print(f"\nGame review distribution:")
        print(f"• Games with 1 review: {(game_counts == 1).sum()}")
        print(f"• Games with 2-5 reviews: {((game_counts >= 2) & (game_counts <= 5)).sum()}")
        print(f"• Games with 6-10 reviews: {((game_counts >= 6) & (game_counts <= 10)).sum()}")
        print(f"• Games with >10 reviews: {(game_counts > 10).sum()}")

        # Check for potential data leakage features
        suspicious_features = []
        for col in enriched_data.columns:
            if 'is_recommended' in col.lower() and col != 'is_recommended':
                suspicious_features.append(col)

        if suspicious_features:
            print(f"\n⚠️ Potential data leakage features found:")
            for feature in suspicious_features:
                print(f"• {feature}")

        # Analyze feature correlations with target
        numeric_cols = enriched_data.select_dtypes(include=[np.number]).columns
        if 'is_recommended' in numeric_cols:
            correlations = enriched_data[numeric_cols].corr()['is_recommended'].abs().sort_values(ascending=False)
            print(f"\nTop 10 features correlated with target:")
            for feature, corr in correlations.head(10).items():
                if feature != 'is_recommended':
                    print(f"• {feature}: {corr:.3f}")

    except Exception as e:
        print(f"Error in analysis: {e}")


def quick_test_with_popular_games():
    """Test recommendations using only popular games"""

    print("🎮 Quick Test with Popular Steam Games")
    print("=" * 40)

    # Popular Steam game IDs (these should exist in most datasets)
    popular_games = {
        'Counter-Strike: Global Offensive': 730,
        'Dota 2': 570,
        'Team Fortress 2': 440,
        'Left 4 Dead 2': 550,
        'Portal 2': 620,
        'Half-Life 2': 220,
        'Garry\'s Mod': 4000,
        'Civilization V': 8930,
        'Skyrim': 72850,
        'GTA V': 271590
    }

    try:
        enriched_data = pd.read_parquet("data/04_feature/enriched_features.parquet")
        available_games = enriched_data['app_id'].unique()

        print("Available popular games in dataset:")
        found_games = []
        for name, app_id in popular_games.items():
            if app_id in available_games:
                print(f"✓ {name} (ID: {app_id})")
                found_games.append(app_id)
            else:
                print(f"✗ {name} (ID: {app_id}) - Not found")

        if len(found_games) >= 2:
            print(f"\n🎯 Testing with games: {found_games[:3]}")
            # You can use these game IDs for testing
        else:
            print("\n⚠️ Not enough popular games found in dataset")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Improved AutoGluon Recommendation Pipeline")
    parser.add_argument("--analyze", action="store_true", help="Analyze current model issues")
    parser.add_argument("--test-games", action="store_true", help="Test with popular games")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full retraining pipeline")

    args = parser.parse_args()

    if args.analyze:
        analyze_current_model_issues()
    elif args.test_games:
        quick_test_with_popular_games()
    elif args.full_pipeline:
        run_complete_pipeline()
    else:
        print("Usage:")
        print("python retrain_pipeline.py --analyze          # Analyze current issues")
        print("python retrain_pipeline.py --test-games       # Check available popular games")
        print("python retrain_pipeline.py --full-pipeline    # Run complete retraining")