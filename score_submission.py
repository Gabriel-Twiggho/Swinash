"""
Score a mock competition submission against answer_key.csv.

In a real Kaggle competition, you would not have the answer key.
For this practice setup, this script acts like a local leaderboard.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent

SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "xgboost_baseline_submission.csv"
ANSWER_KEY_PATH = Path(r"D:/Documents/Hackathon/answer_key.csv")

REPORT_DIR = PROJECT_ROOT / "reports" / "submission_score"
SCORE_REPORT_PATH = REPORT_DIR / "score_report.csv"

ID_COLUMN = "Id"
TARGET_RETURN = "Target_Return"
TARGET_DIRECTION = "Target_Direction"
REQUIRED_COLUMNS = [ID_COLUMN, TARGET_RETURN, TARGET_DIRECTION]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name} at {path}")
    return pd.read_csv(path)


def check_required_columns(df: pd.DataFrame, name: str) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")


def clean_submission(submission: pd.DataFrame) -> pd.DataFrame:
    submission = submission[REQUIRED_COLUMNS].copy()

    submission[ID_COLUMN] = pd.to_numeric(submission[ID_COLUMN], errors="coerce")
    submission[TARGET_RETURN] = pd.to_numeric(submission[TARGET_RETURN], errors="coerce")
    submission[TARGET_DIRECTION] = pd.to_numeric(submission[TARGET_DIRECTION], errors="coerce")

    if submission[ID_COLUMN].isna().any():
        raise ValueError("Submission has missing or invalid IDs.")

    if submission[TARGET_RETURN].isna().any():
        raise ValueError("Submission has missing or invalid Target_Return values.")

    if submission[TARGET_DIRECTION].isna().any():
        raise ValueError("Submission has missing or invalid Target_Direction values.")

    submission[ID_COLUMN] = submission[ID_COLUMN].astype(int)
    submission[TARGET_DIRECTION] = submission[TARGET_DIRECTION].astype(int)

    invalid_directions = ~submission[TARGET_DIRECTION].isin([0, 1])
    if invalid_directions.any():
        raise ValueError("Submission Target_Direction must only contain 0 or 1.")

    duplicate_ids = submission[ID_COLUMN].duplicated().sum()
    if duplicate_ids:
        raise ValueError(f"Submission has {duplicate_ids:,} duplicate IDs.")

    return submission


def clean_answer_key(answer_key: pd.DataFrame) -> pd.DataFrame:
    answer_key = answer_key[REQUIRED_COLUMNS].copy()

    answer_key[ID_COLUMN] = pd.to_numeric(answer_key[ID_COLUMN], errors="coerce").astype(int)
    answer_key[TARGET_RETURN] = pd.to_numeric(answer_key[TARGET_RETURN], errors="coerce")
    answer_key[TARGET_DIRECTION] = pd.to_numeric(answer_key[TARGET_DIRECTION], errors="coerce").astype(int)

    return answer_key


def score_submission(submission: pd.DataFrame, answer_key: pd.DataFrame) -> dict[str, float]:
    scored = answer_key.merge(
        submission,
        on=ID_COLUMN,
        how="left",
        suffixes=("_actual", "_predicted"),
    )

    missing_predictions = scored[f"{TARGET_RETURN}_predicted"].isna().sum()
    if missing_predictions:
        raise ValueError(f"Submission is missing predictions for {missing_predictions:,} answer key rows.")

    actual_returns = scored[f"{TARGET_RETURN}_actual"]
    predicted_returns = scored[f"{TARGET_RETURN}_predicted"]
    actual_directions = scored[f"{TARGET_DIRECTION}_actual"]
    predicted_directions = scored[f"{TARGET_DIRECTION}_predicted"]

    rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))
    mae = mean_absolute_error(actual_returns, predicted_returns)
    direction_accuracy = accuracy_score(actual_directions, predicted_directions)

    implied_directions = (predicted_returns > 0).astype(int)
    direction_consistency = accuracy_score(implied_directions, predicted_directions)

    return {
        "rows_scored": len(scored),
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_accuracy,
        "submission_direction_matches_return_sign": direction_consistency,
    }


def save_score_report(metrics: dict[str, float]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(SCORE_REPORT_PATH, index=False)
    print(f"Saved: {SCORE_REPORT_PATH}")


def main() -> None:
    print_section("Score Submission")
    print(f"Submission: {SUBMISSION_PATH}")
    print(f"Answer key: {ANSWER_KEY_PATH}")

    submission = load_csv(SUBMISSION_PATH, "submission")
    answer_key = load_csv(ANSWER_KEY_PATH, "answer_key.csv")

    check_required_columns(submission, "submission")
    check_required_columns(answer_key, "answer_key.csv")

    submission = clean_submission(submission)
    answer_key = clean_answer_key(answer_key)

    metrics = score_submission(submission, answer_key)

    print_section("Score")
    print(f"Rows scored: {metrics['rows_scored']:,}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Direction accuracy: {metrics['direction_accuracy']:.4f}")
    print(
        "Submission direction matches return sign: "
        f"{metrics['submission_direction_matches_return_sign']:.4f}"
    )

    save_score_report(metrics)

    print_section("Conclusion")
    print("Local scoring complete.")
    print("In a real competition, the answer key would be hidden by Kaggle.")


if __name__ == "__main__":
    main()
