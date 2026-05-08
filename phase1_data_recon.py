"""
Phase 1 — data recon for the hackathon CSVs.

Loads train / test / sample_submission and prints shapes, columns, missing values,
time ordering, target summaries, and expected submission layout. Run before modeling.
"""
from pathlib import Path

import pandas as pd

# Absolute paths to competition files (adjust if you move the data).
TRAIN_PATH = Path(r"D:/Documents/Hackathon/train.csv")
TEST_PATH = Path(r"D:/Documents/Hackathon/test.csv")
SAMPLE_SUBMISSION_PATH = Path(r"D:/Documents/Hackathon/sample_submission.csv")

# Labels to predict; everything else in train.csv is treated as a feature.
TARGET_COLUMNS = ["Target_Return", "Target_Direction"]


def print_section(title: str) -> None:
    """Visual separator in stdout so logs are easy to scan."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name} at {path}")
    return pd.read_csv(path)


def report_shape(df: pd.DataFrame, name: str) -> None:
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")


def report_columns(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} columns:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:>2}. {column}")


def report_missing_values(df: pd.DataFrame, name: str) -> None:
    # Only list columns with at least one NA to keep output short.
    missing = df.isna().sum()
    missing = missing[missing > 0]

    print(f"\n{name} missing values:")
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())


def report_time_series(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("Time-Series Check")

    if "Date" not in train.columns or "Date" not in test.columns:
        print("No Date column found, so this does not look like obvious time-series data.")
        return

    train_dates = pd.to_datetime(train["Date"])
    test_dates = pd.to_datetime(test["Date"])

    print(f"Train date range: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"Test date range:  {test_dates.min().date()} to {test_dates.max().date()}")
    print(f"Train dates sorted: {train_dates.is_monotonic_increasing}")
    print(f"Test dates sorted:  {test_dates.is_monotonic_increasing}")
    # True usually means test is a future holdout — avoid shuffled CV if so.
    print(f"Test starts after train ends: {test_dates.min() > train_dates.max()}")

    print("\nConclusion: this is time-series data, so do not use a random train/test split.")


def report_targets(train: pd.DataFrame) -> None:
    print_section("Target Check")

    for target in TARGET_COLUMNS:
        if target not in train.columns:
            print(f"{target}: missing from train.csv")
            continue

        unique_values = train[target].nunique()
        dtype = train[target].dtype
        print(f"{target}: dtype={dtype}, unique values={unique_values}")

        if target == "Target_Return":
            print("  Type: regression target because it is a continuous numeric value.")
            print(
                "  Summary:",
                train[target].describe()[["mean", "std", "min", "max"]].round(6).to_dict(),
            )

        if target == "Target_Direction":
            print("  Type: classification target because it is a 0/1 label.")
            print("  Class balance:", train[target].value_counts().sort_index().to_dict())


def report_submission_format(sample_submission: pd.DataFrame) -> None:
    print_section("Submission Format")
    print("Expected columns:", sample_submission.columns.tolist())
    print(f"Expected submission rows: {len(sample_submission):,}")
    print("\nFirst few rows:")
    print(sample_submission.head().to_string(index=False))


def main() -> None:
    train = load_csv(TRAIN_PATH, "train.csv")
    test = load_csv(TEST_PATH, "test.csv")
    sample_submission = load_csv(SAMPLE_SUBMISSION_PATH, "sample_submission.csv")

    print_section("Phase 1: Data Recon")
    report_shape(train, "train.csv")
    report_shape(test, "test.csv")
    report_shape(sample_submission, "sample_submission.csv")

    # Features = all train columns except the two target fields.
    print_section("Feature Names")
    feature_columns = [column for column in train.columns if column not in TARGET_COLUMNS]
    print("Feature columns used for prediction:")
    for index, column in enumerate(feature_columns, start=1):
        print(f"{index:>2}. {column}")

    report_columns(train, "train.csv")
    report_missing_values(train, "train.csv")
    report_missing_values(test, "test.csv")
    report_time_series(train, test)
    report_targets(train)
    report_submission_format(sample_submission)

    print_section("Phase 1 Conclusion")
    print("You now know:")
    print("- how many rows and columns each file has")
    print("- what features are available")
    print("- whether missing values need immediate attention")
    print("- that the data is chronological time-series data")
    print("- that Target_Return is regression and Target_Direction is classification")


if __name__ == "__main__":
    main()
