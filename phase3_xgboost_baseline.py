"""
Phase 3: XGBoost Baseline

Train a simple XGBoost model to predict Target_Return.
Then derive Target_Direction from the predicted return:
- predicted return > 0  -> direction 1
- predicted return <= 0 -> direction 0

This script creates a valid submission CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed. Run `uv sync`, then run this script again."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent

MODELING_DIR = PROJECT_ROOT / "data" / "modeling"
TRAIN_FEATURES_PATH = MODELING_DIR / "train_features.csv"
TEST_FEATURES_PATH = MODELING_DIR / "test_features.csv"
TRAIN_TARGET_RETURN_PATH = MODELING_DIR / "train_target_return.csv"
TRAIN_TARGET_DIRECTION_PATH = MODELING_DIR / "train_target_direction.csv"
TRAIN_IDS_PATH = MODELING_DIR / "train_ids.csv"
SAMPLE_SUBMISSION_PATH = MODELING_DIR / "sample_submission.csv"

SUBMISSION_DIR = PROJECT_ROOT / "submissions"

SUBMISSION_PATH = SUBMISSION_DIR / "xgboost_baseline_submission.csv"

TARGET_RETURN = "Target_Return"
TARGET_DIRECTION = "Target_Direction"
VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {name} at {path}. Run phase2_5_prepare_features.py first."
        )
    return pd.read_csv(path)


def split_by_order(
    X: pd.DataFrame,
    y_return: pd.Series,
    y_direction: pd.Series,
    ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    split_index = int(len(X) * (1 - VALIDATION_FRACTION))

    X_train = X.iloc[:split_index]
    X_val = X.iloc[split_index:]

    y_return_train = y_return.iloc[:split_index]
    y_return_val = y_return.iloc[split_index:]

    y_direction_train = y_direction.iloc[:split_index]
    y_direction_val = y_direction.iloc[split_index:]

    train_ids = ids.iloc[:split_index]
    val_ids = ids.iloc[split_index:]

    return (
        X_train,
        X_val,
        y_return_train,
        y_return_val,
        y_direction_train,
        y_direction_val,
        train_ids,
        val_ids,
    )


def build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def direction_from_return(predicted_returns: np.ndarray) -> np.ndarray:
    return (predicted_returns > 0).astype(int)


def evaluate_validation(
    y_return_val: pd.Series,
    y_direction_val: pd.Series,
    predicted_returns: np.ndarray,
) -> dict[str, float]:
    predicted_directions = direction_from_return(predicted_returns)

    rmse = np.sqrt(mean_squared_error(y_return_val, predicted_returns))
    mae = mean_absolute_error(y_return_val, predicted_returns)
    direction_accuracy = accuracy_score(y_direction_val, predicted_directions)

    return {
        "validation_rmse": rmse,
        "validation_mae": mae,
        "validation_direction_accuracy": direction_accuracy,
    }


def print_feature_importance(model: XGBRegressor, feature_names: list[str]) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    importance = importance.sort_values("importance", ascending=False)

    print("\nTop feature importances:")
    for row in importance.head(10).itertuples(index=False):
        print(f"- {row.feature}: {row.importance:.4f}")


def create_submission(
    model: XGBRegressor,
    test_features: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    predicted_returns = model.predict(test_features)

    submission = sample_submission.copy()
    submission[TARGET_RETURN] = predicted_returns
    submission[TARGET_DIRECTION] = direction_from_return(predicted_returns)

    return submission


def save_submission(submission: pd.DataFrame) -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved: {SUBMISSION_PATH}")


def main() -> None:
    train_features = load_csv(TRAIN_FEATURES_PATH, "train_features.csv")
    test_features = load_csv(TEST_FEATURES_PATH, "test_features.csv")
    y_return = load_csv(TRAIN_TARGET_RETURN_PATH, "train_target_return.csv")[TARGET_RETURN]
    y_direction = load_csv(TRAIN_TARGET_DIRECTION_PATH, "train_target_direction.csv")[TARGET_DIRECTION]
    train_ids = load_csv(TRAIN_IDS_PATH, "train_ids.csv")["Id"]
    sample_submission = load_csv(SAMPLE_SUBMISSION_PATH, "sample_submission.csv")

    print_section("Phase 3: XGBoost Baseline")
    print(f"Train feature rows: {len(train_features):,}")
    print(f"Test feature rows: {len(test_features):,}")
    print(f"Feature count: {train_features.shape[1]}")
    print(f"Validation fraction: {VALIDATION_FRACTION:.0%}")

    (
        X_train,
        X_val,
        y_return_train,
        y_return_val,
        _y_direction_train,
        y_direction_val,
        _train_ids,
        val_ids,
    ) = split_by_order(train_features, y_return, y_direction, train_ids)

    print_section("Train Validation Model")
    model = build_model()
    model.fit(X_train, y_return_train)
    val_predictions = model.predict(X_val)
    metrics = evaluate_validation(y_return_val, y_direction_val, val_predictions)

    print(f"Validation rows: {len(X_val):,}")
    print(f"Validation ID range: {int(val_ids.min())} to {int(val_ids.max())}")
    print(f"RMSE: {metrics['validation_rmse']:.6f}")
    print(f"MAE: {metrics['validation_mae']:.6f}")
    print(f"Direction accuracy: {metrics['validation_direction_accuracy']:.4f}")

    print_feature_importance(model, train_features.columns.tolist())

    print_section("Train Final Model")
    final_model = build_model()
    final_model.fit(train_features, y_return)

    submission = create_submission(final_model, test_features, sample_submission)
    save_submission(submission)

    print_section("Phase 3 Conclusion")
    print("XGBoost baseline is complete.")
    print("Submission file is ready to upload or score against the mock answer key.")


if __name__ == "__main__":
    main()
