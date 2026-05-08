"""
Phase 1.5: Clean Data

Read the raw files, fix the obvious issues found in Phase 1, and save cleaned
CSV files into data/processed/.

For this practice dataset, Id is treated as the reliable time order. The Date
column is checked, but not used for the first model.

The raw files are never overwritten.
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_TRAIN_PATH = Path(r"D:/Documents/Hackathon/train.csv")
RAW_TEST_PATH = Path(r"D:/Documents/Hackathon/test.csv")
RAW_SAMPLE_SUBMISSION_PATH = Path(r"D:/Documents/Hackathon/sample_submission.csv")

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_TRAIN_PATH = OUTPUT_DIR / "train_clean.csv"
CLEAN_TEST_PATH = OUTPUT_DIR / "test_clean.csv"
CLEAN_SAMPLE_SUBMISSION_PATH = OUTPUT_DIR / "sample_submission_clean.csv"

ID_COLUMN = "Id"
DATE_COLUMN = "Date"
TARGET_RETURN = "Target_Return"
TARGET_DIRECTION = "Target_Direction"
TARGET_COLUMNS = [TARGET_RETURN, TARGET_DIRECTION]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name} at {path}")
    return pd.read_csv(path)


def clean_number_column(series: pd.Series) -> pd.Series:
    """Convert values like '1,660,275' into real numbers."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def is_mostly_numeric(series: pd.Series) -> bool:
    numeric = clean_number_column(series)
    return numeric.notna().mean() >= 0.8


def choose_numeric_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    features = []

    for column in train.columns:
        if column in TARGET_COLUMNS or column in [ID_COLUMN, DATE_COLUMN]:
            continue

        if column not in test.columns:
            continue

        if is_mostly_numeric(train[column]) and is_mostly_numeric(test[column]):
            features.append(column)

    return features


def clean_ids(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_section("Clean IDs")

    train = train.copy()
    test = test.copy()
    sample_submission = sample_submission.copy()

    missing_test_ids = test[ID_COLUMN].isna()
    if missing_test_ids.any() and len(test) == len(sample_submission):
        test.loc[missing_test_ids, ID_COLUMN] = sample_submission.loc[missing_test_ids, ID_COLUMN]
        print(f"Filled missing test IDs from sample_submission: {missing_test_ids.sum():,}")

    train[ID_COLUMN] = clean_number_column(train[ID_COLUMN])
    test[ID_COLUMN] = clean_number_column(test[ID_COLUMN])
    sample_submission[ID_COLUMN] = clean_number_column(sample_submission[ID_COLUMN])

    before = len(train)
    train = train.dropna(subset=[ID_COLUMN, TARGET_RETURN])
    print(f"Dropped train rows missing ID or Target_Return: {before - len(train):,}")

    before = len(train)
    train = train.drop_duplicates(subset=[ID_COLUMN], keep="first")
    print(f"Dropped duplicate train IDs: {before - len(train):,}")

    if test[ID_COLUMN].isna().any():
        raise ValueError("test.csv still has missing IDs after cleaning.")

    train[ID_COLUMN] = train[ID_COLUMN].astype(int)
    test[ID_COLUMN] = test[ID_COLUMN].astype(int)
    sample_submission[ID_COLUMN] = sample_submission[ID_COLUMN].astype(int)

    return train, test, sample_submission


def clean_targets(train: pd.DataFrame) -> pd.DataFrame:
    print_section("Clean Targets")

    train = train.copy()
    train[TARGET_RETURN] = clean_number_column(train[TARGET_RETURN])

    before = len(train)
    train = train.dropna(subset=[TARGET_RETURN])
    print(f"Dropped rows missing Target_Return: {before - len(train):,}")

    # For this beginner strategy, direction is whether return is positive.
    train[TARGET_DIRECTION] = (train[TARGET_RETURN] > 0).astype(int)

    print("Target_Direction values after cleaning:")
    print(train[TARGET_DIRECTION].value_counts().sort_index().to_string())

    return train


def clean_dates(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("Check Dates")

    train = train.copy()
    test = test.copy()

    train_dates = pd.to_datetime(train[DATE_COLUMN], format="mixed", errors="coerce")
    test_dates = pd.to_datetime(test[DATE_COLUMN], format="mixed", errors="coerce")

    print(f"Invalid train dates: {train_dates.isna().sum():,}")
    print(f"Invalid test dates: {test_dates.isna().sum():,}")
    print(f"Train dates sorted: {train_dates.dropna().is_monotonic_increasing}")
    print(f"Test dates sorted: {test_dates.dropna().is_monotonic_increasing}")
    print("Keeping Date for reference, but using Id order for the first modeling workflow.")

    return train, test


def clean_numeric_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("Clean Numeric Features")

    train = train.copy()
    test = test.copy()

    print("Features kept for the first model:")
    for index, column in enumerate(numeric_features, start=1):
        print(f"{index:>2}. {column}")

    for column in numeric_features:
        train[column] = clean_number_column(train[column])
        test[column] = clean_number_column(test[column])

    train_medians = train[numeric_features].median()
    train[numeric_features] = train[numeric_features].ffill().fillna(train_medians)
    test[numeric_features] = test[numeric_features].ffill().fillna(train_medians)

    print(f"Missing numeric cells in clean train: {train[numeric_features].isna().sum().sum():,}")
    print(f"Missing numeric cells in clean test: {test[numeric_features].isna().sum().sum():,}")

    return train, test


def keep_model_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_columns = [ID_COLUMN, DATE_COLUMN] + numeric_features + TARGET_COLUMNS
    test_columns = [ID_COLUMN, DATE_COLUMN] + numeric_features

    train = train[train_columns].sort_values(ID_COLUMN).reset_index(drop=True)
    test = test[test_columns].sort_values(ID_COLUMN).reset_index(drop=True)
    sample_submission = sample_submission.sort_values(ID_COLUMN).reset_index(drop=True)

    return train, test, sample_submission


def save_clean_files(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(CLEAN_TRAIN_PATH, index=False)
    test.to_csv(CLEAN_TEST_PATH, index=False)
    sample_submission.to_csv(CLEAN_SAMPLE_SUBMISSION_PATH, index=False)


def report_outputs(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    print_section("Cleaned Output")

    print(f"Saved: {CLEAN_TRAIN_PATH}")
    print(f"Saved: {CLEAN_TEST_PATH}")
    print(f"Saved: {CLEAN_SAMPLE_SUBMISSION_PATH}")

    print(f"\ntrain_clean.csv: {train.shape[0]:,} rows x {train.shape[1]:,} columns")
    print(f"test_clean.csv: {test.shape[0]:,} rows x {test.shape[1]:,} columns")
    print(
        "sample_submission_clean.csv: "
        f"{sample_submission.shape[0]:,} rows x {sample_submission.shape[1]:,} columns"
    )

    print(f"\nMissing cells in clean train: {train.isna().sum().sum():,}")
    print(f"Missing cells in clean test: {test.isna().sum().sum():,}")


def main() -> None:
    train = load_csv(RAW_TRAIN_PATH, "train.csv")
    test = load_csv(RAW_TEST_PATH, "test.csv")
    sample_submission = load_csv(RAW_SAMPLE_SUBMISSION_PATH, "sample_submission.csv")

    print_section("Phase 1.5: Clean Data")
    print("Raw files are kept unchanged.")
    print(f"Raw train shape: {train.shape[0]:,} rows x {train.shape[1]:,} columns")
    print(f"Raw test shape: {test.shape[0]:,} rows x {test.shape[1]:,} columns")

    train, test, sample_submission = clean_ids(train, test, sample_submission)
    train = clean_targets(train)
    train, test = clean_dates(train, test)

    numeric_features = choose_numeric_features(train, test)
    train, test = clean_numeric_features(train, test, numeric_features)

    train, test, sample_submission = keep_model_columns(
        train,
        test,
        sample_submission,
        numeric_features,
    )

    save_clean_files(train, test, sample_submission)
    report_outputs(train, test, sample_submission)

    print_section("Phase 1.5 Conclusion")
    print("Cleaned files are ready.")
    print(f"First model will use {len(numeric_features)} numeric features.")
    print("Next step: run Phase 2 visual EDA, then Phase 2.5 feature preparation.")


if __name__ == "__main__":
    main()
