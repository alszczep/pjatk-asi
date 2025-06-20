import pandas as pd
import os
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def train_autogluon_recommender(training_data: pd.DataFrame) -> str:
    model_path = os.path.abspath("data/06_models/autogluon_recommender")
    os.makedirs(model_path, exist_ok=True)

    training_sample = training_data.copy()
    if len(training_data) > 15000:
        positive_samples = training_data[training_data['is_recommended'] == True]
        negative_samples = training_data[training_data['is_recommended'] == False]

        pos_ratio = len(positive_samples) / len(training_data)
        n_pos = int(15000 * pos_ratio)
        n_neg = 15000 - n_pos

        pos_sample = positive_samples.sample(n=min(n_pos, len(positive_samples)), random_state=42)
        neg_sample = negative_samples.sample(n=min(n_neg, len(negative_samples)), random_state=42)

        training_sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)

    training_sample = training_sample.fillna(0)

    predictor = TabularPredictor(
        label='is_recommended',
        path=model_path,
        problem_type='binary',
        eval_metric='roc_auc'
    ).fit(
        training_sample,
        presets='best_quality',
        verbosity=1,
        ag_args_fit={
            'num_bag_folds': 3,
            'num_stack_levels': 1
        }
    )

    print("AutoGluon training completed!")

    path_file = "data/06_models/autogluon_model_path.txt"
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, 'w') as f:
        f.write(model_path)

    return model_path


def evaluate_recommender(model_path: str, training_data: pd.DataFrame) -> pd.DataFrame:
    predictor = TabularPredictor.load(model_path)

    if 'is_recommended' not in training_data.columns:
        raise ValueError(
            f"Column 'is_recommended' not found in training_data. Available columns: {training_data.columns.tolist()}")

    test_size = min(2000, len(training_data) // 3)
    test_data = training_data.sample(n=test_size, random_state=123)

    feature_cols = [col for col in test_data.columns if col != 'is_recommended']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['is_recommended']

    print(f"Test set size: {len(test_data)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target distribution: {y_test.value_counts()}")

    try:
        y_pred = predictor.predict(X_test)
        y_pred_proba = predictor.predict_proba(X_test)

        if hasattr(y_pred_proba, 'iloc') and y_pred_proba.shape[1] > 1:
            proba_positive = y_pred_proba.iloc[:, 1]
        elif hasattr(y_pred_proba, 'values') and len(y_pred_proba.shape) > 1:
            proba_positive = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
        else:
            proba_positive = y_pred_proba

        auc_score = roc_auc_score(y_test, proba_positive)
        report = classification_report(y_test, y_pred, output_dict=True)

        eval_df = pd.DataFrame({
            'metric': ['accuracy', 'precision', 'recall', 'f1-score', 'auc'],
            'value': [
                report['accuracy'],
                report[True]['precision'] if True in report else report['True']['precision'],
                report[True]['recall'] if True in report else report['True']['recall'],
                report[True]['f1-score'] if True in report else report['True']['f1-score'],
                auc_score
            ]
        })

        print("=== AutoGluon Recommender Evaluation ===")
        print(eval_df)

        try:
            feature_importance = predictor.feature_importance(X_test)
            print("\n=== Top 10 Feature Importance ===")
            print(feature_importance.head(10))
        except Exception as e:
            print(f"Could not calculate feature importance: {e}")

        return eval_df

    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Trying alternative approach...")

        try:

            test_data_with_target = test_data.copy()
            results = predictor.evaluate(test_data_with_target, silent=False)

            eval_df = pd.DataFrame({
                'metric': list(results.keys()),
                'value': list(results.values())
            })

            print("=== AutoGluon Recommender Evaluation (Alternative) ===")
            print(eval_df)

            return eval_df

        except Exception as e2:
            print(f"Alternative evaluation also failed: {e2}")
            raise
