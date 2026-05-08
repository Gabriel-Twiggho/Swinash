"""
Phase 1: Data Recon

Read the raw competition files and print the key checks before modeling:
- file sizes
- columns and possible features
- missing values
- train/test feature mismatch
- date ordering
- target summaries
- submission format

This script only reports. It does not clean or save data.
"""
from pathlib import Path

import pandas as pd


TRAIN_PATH = Path(r"D:/Documents/Hackathon/train.csv")
TEST_PATH = Path(r"D:/Documents/Hackathon/test.csv")
SAMPLE_SUBMISSION_PATH = Path(r"D:/Documents/Hackathon/sample_submission.csv")

TARGET_COLUMNS = ["Target_Return", "Target_Direction"]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name} at {path}")
    return pd.read_csv(path)


def report_shapes(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    print_section("File Sizes")
    print(f"train.csv: {train.shape[0]:,} rows x {train.shape[1]:,} columns")
    print(f"test.csv: {test.shape[0]:,} rows x {test.shape[1]:,} columns")
    print(f"sample_submission.csv: {sample.shape[0]:,} rows x {sample.shape[1]:,} columns")


def report_columns(train: pd.DataFrame) -> None:
    print_section("Columns")

    print("All train columns:")
    for index, column in enumerate(train.columns, start=1):
        print(f"{index:>2}. {column}")

    feature_columns = [column for column in train.columns if column not in TARGET_COLUMNS]

    print("\nPossible feature columns:")
    for index, column in enumerate(feature_columns, start=1):
        print(f"{index:>2}. {column}")


def report_missing_values(df: pd.DataFrame, name: str) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]

    print(f"\n{name} missing values:")
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())


def report_train_test_feature_match(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("Train/Test Feature Match")

    train_features = [column for column in train.columns if column not in TARGET_COLUMNS]
    test_features = list(test.columns)

    train_only = [column for column in train_features if column not in test_features]
    test_only = [column for column in test_features if column not in train_features]

    if not train_only and not test_only:
        print("Train and test feature columns match.")
        return

    if train_only:
        print("Columns in train but not test:")
        for column in train_only:
            print(f"- {column}")

    if test_only:
        print("\nColumns in test but not train:")
        for column in test_only:
            print(f"- {column}")

    print("\nThese columns need a decision before modeling.")


def clean_number_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def report_id_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("ID Order Check")

    if "Id" not in train.columns or "Id" not in test.columns:
        print("No Id column found in both train and test.")
        return

    train_ids = clean_number_column(train["Id"])
    test_ids = clean_number_column(test["Id"])

    print(f"Train missing IDs: {train_ids.isna().sum():,}")
    print(f"Test missing IDs: {test_ids.isna().sum():,}")
    print(f"Train duplicate IDs: {train_ids.duplicated().sum():,}")
    print(f"Test duplicate IDs: {test_ids.duplicated().sum():,}")

    valid_train_ids = train_ids.dropna()
    valid_test_ids = test_ids.dropna()

    if valid_train_ids.empty or valid_test_ids.empty:
        print("Not enough valid IDs to check ordering.")
        return

    print(f"Train ID range: {int(valid_train_ids.min())} to {int(valid_train_ids.max())}")
    print(f"Test ID range:  {int(valid_test_ids.min())} to {int(valid_test_ids.max())}")
    print(f"Train IDs sorted: {valid_train_ids.is_monotonic_increasing}")
    print(f"Test IDs sorted:  {valid_test_ids.is_monotonic_increasing}")
    print(f"Test starts after train by ID: {valid_test_ids.min() > valid_train_ids.max()}")

    print("\nIf Date looks messy, ID order is the safer time order for this practice dataset.")


def parse_dates_safely(date_column: pd.Series, name: str) -> pd.Series:
    dates = pd.to_datetime(date_column, format="mixed", errors="coerce")
    invalid_dates = dates.isna()

    if invalid_dates.any():
        print(f"\n{name} invalid Date values: {invalid_dates.sum():,}")
        print("Examples:")
        print(date_column[invalid_dates].head(5).to_string(index=False))

    return dates


def report_time_series(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("Time-Series Check")

    if "Date" not in train.columns or "Date" not in test.columns:
        print("No Date column found in both train and test.")
        return

    train_dates = parse_dates_safely(train["Date"], "train.csv")
    test_dates = parse_dates_safely(test["Date"], "test.csv")

    valid_train_dates = train_dates.dropna()
    valid_test_dates = test_dates.dropna()

    if valid_train_dates.empty or valid_test_dates.empty:
        print("Not enough valid dates to check time ordering.")
        return

    print(f"Train date range: {valid_train_dates.min().date()} to {valid_train_dates.max().date()}")
    print(f"Test date range:  {valid_test_dates.min().date()} to {valid_test_dates.max().date()}")
    print(f"Train dates sorted: {valid_train_dates.is_monotonic_increasing}")
    print(f"Test dates sorted:  {valid_test_dates.is_monotonic_increasing}")
    print(f"Test starts after train ends: {valid_test_dates.min() > valid_train_dates.max()}")
    print("\nReminder: this is time-series data, so do not randomly split rows.")


def report_targets(train: pd.DataFrame) -> None:
    print_section("Target Check")

    if "Target_Return" in train.columns:
        print("Target_Return: regression target")
        print(train["Target_Return"].describe()[["mean", "std", "min", "max"]].round(6).to_string())
    else:
        print("Target_Return is missing from train.csv")

    if "Target_Direction" in train.columns:
        print("\nTarget_Direction: classification target")
        print(train["Target_Direction"].value_counts(dropna=False).sort_index().to_string())
    else:
        print("\nTarget_Direction is missing from train.csv")


def report_submission_format(sample_submission: pd.DataFrame) -> None:
    print_section("Submission Format")
    print("Expected columns:", sample_submission.columns.tolist())
    print(f"Expected rows: {len(sample_submission):,}")
    print("\nFirst few rows:")
    print(sample_submission.head().to_string(index=False))


def main() -> None:
    train = load_csv(TRAIN_PATH, "train.csv")
    test = load_csv(TEST_PATH, "test.csv")
    sample_submission = load_csv(SAMPLE_SUBMISSION_PATH, "sample_submission.csv")

    print_section("Phase 1: Data Recon")
    report_shapes(train, test, sample_submission)
    report_columns(train)

    print_section("Missing Values")
    report_missing_values(train, "train.csv")
    report_missing_values(test, "test.csv")

    report_train_test_feature_match(train, test)
    report_id_order(train, test)
    report_time_series(train, test)
    report_targets(train)
    report_submission_format(sample_submission)

    print_section("Phase 1 Conclusion")
    print("You now know what looks dirty or unusual before modeling.")
    print("Next step: run Phase 1.5 to create cleaned files.")


if __name__ == "__main__":
    main()
