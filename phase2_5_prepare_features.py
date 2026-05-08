"""
Phase 2.5: Prepare Features

Take the cleaned data and create simple modeling files for Phase 3.

This script can:
- choose shared numeric features
- encode shared categorical features if any exist
- save outlier cap suggestions
- optionally cap outliers
- save X/y style files for the first model

This script does not train a model.
It does not standardise/normalise features. That belongs in Phase 3, after the
train/validation split, so we avoid leakage.
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_clean.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "data" / "processed" / "sample_submission_clean.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "modeling"

TRAIN_FEATURES_PATH = OUTPUT_DIR / "train_features.csv"
TEST_FEATURES_PATH = OUTPUT_DIR / "test_features.csv"
TRAIN_TARGET_RETURN_PATH = OUTPUT_DIR / "train_target_return.csv"
TRAIN_TARGET_DIRECTION_PATH = OUTPUT_DIR / "train_target_direction.csv"
TRAIN_IDS_PATH = OUTPUT_DIR / "train_ids.csv"
TEST_IDS_PATH = OUTPUT_DIR / "test_ids.csv"
SAMPLE_SUBMISSION_PATH_OUT = OUTPUT_DIR / "sample_submission.csv"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.csv"

ID_COLUMN = "Id"
DATE_COLUMN = "Date"
TARGET_RETURN = "Target_Return"
TARGET_DIRECTION = "Target_Direction"
TARGET_COLUMNS = [TARGET_RETURN, TARGET_DIRECTION]

# Keep False for the beginner first pass. We discover outliers, but do not
# change them unless we choose to.
CAP_OUTLIERS = False


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {name} at {path}. Run phase1_5_clean_data.py first."
        )
    return pd.read_csv(path)


def get_shared_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    features = []

    for column in train.columns:
        if column in TARGET_COLUMNS or column in [ID_COLUMN, DATE_COLUMN]:
            continue

        if column in test.columns:
            features.append(column)

    return features


def split_feature_types(
    train: pd.DataFrame,
    shared_features: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = []
    categorical_features = []

    for column in shared_features:
        if pd.api.types.is_numeric_dtype(train[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


def encode_categorical_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not categorical_features:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)

    combined = pd.concat(
        [train[categorical_features], test[categorical_features]],
        axis=0,
        ignore_index=True,
    )
    encoded = pd.get_dummies(combined, columns=categorical_features, dummy_na=True)

    train_encoded = encoded.iloc[: len(train)].reset_index(drop=True)
    test_encoded = encoded.iloc[len(train) :].reset_index(drop=True)

    return train_encoded, test_encoded


def get_outlier_caps(train: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows = []

    for feature in numeric_features:
        values = train[feature]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_cap = q1 - 1.5 * iqr
        upper_cap = q3 + 1.5 * iqr
        outlier_count = ((values < lower_cap) | (values > upper_cap)).sum()

        rows.append(
            {
                "feature": feature,
                "lower_cap": lower_cap,
                "upper_cap": upper_cap,
                "train_outlier_count": outlier_count,
                "train_outlier_percent": outlier_count / len(train),
            }
        )

    return pd.DataFrame(rows).sort_values("train_outlier_percent", ascending=False)


def apply_outlier_caps(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outlier_caps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for row in outlier_caps.itertuples(index=False):
        train[row.feature] = train[row.feature].clip(row.lower_cap, row.upper_cap)
        test[row.feature] = test[row.feature].clip(row.lower_cap, row.upper_cap)

    return train, test


def build_feature_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_numeric = train[numeric_features].reset_index(drop=True)
    test_numeric = test[numeric_features].reset_index(drop=True)

    train_encoded, test_encoded = encode_categorical_features(
        train,
        test,
        categorical_features,
    )

    train_features = pd.concat([train_numeric, train_encoded], axis=1)
    test_features = pd.concat([test_numeric, test_encoded], axis=1)

    return train_features, test_features


def save_feature_files(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(TRAIN_FEATURES_PATH, index=False)
    test_features.to_csv(TEST_FEATURES_PATH, index=False)
    train[[TARGET_RETURN]].to_csv(TRAIN_TARGET_RETURN_PATH, index=False)
    train[[TARGET_DIRECTION]].to_csv(TRAIN_TARGET_DIRECTION_PATH, index=False)
    train[[ID_COLUMN]].to_csv(TRAIN_IDS_PATH, index=False)
    test[[ID_COLUMN]].to_csv(TEST_IDS_PATH, index=False)
    sample_submission.to_csv(SAMPLE_SUBMISSION_PATH_OUT, index=False)

    feature_columns = pd.DataFrame({"feature": train_features.columns})
    feature_columns.to_csv(FEATURE_COLUMNS_PATH, index=False)


def report_outputs(
    numeric_features: list[str],
    categorical_features: list[str],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    outlier_caps: pd.DataFrame,
) -> None:
    print_section("Feature Preparation Summary")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features encoded: {len(categorical_features)}")
    print(f"Final model feature count: {train_features.shape[1]}")
    print(f"Train feature rows: {train_features.shape[0]:,}")
    print(f"Test feature rows: {test_features.shape[0]:,}")
    print(f"Outlier capping applied: {CAP_OUTLIERS}")

    if categorical_features:
        print("\nCategorical columns encoded:")
        for column in categorical_features:
            print(f"- {column}")
    else:
        print("\nNo shared categorical columns found to encode.")

    print("\nTop outlier candidates:")
    for row in outlier_caps.head(10).itertuples(index=False):
        print(f"- {row.feature}: {row.train_outlier_percent:.2%} outliers")

    print_section("Saved Files")
    print(f"Saved: {TRAIN_FEATURES_PATH}")
    print(f"Saved: {TEST_FEATURES_PATH}")
    print(f"Saved: {TRAIN_TARGET_RETURN_PATH}")
    print(f"Saved: {TRAIN_TARGET_DIRECTION_PATH}")
    print(f"Saved: {TRAIN_IDS_PATH}")
    print(f"Saved: {TEST_IDS_PATH}")
    print(f"Saved: {SAMPLE_SUBMISSION_PATH_OUT}")
    print(f"Saved: {FEATURE_COLUMNS_PATH}")


def main() -> None:
    train = load_csv(TRAIN_PATH, "train_clean.csv")
    test = load_csv(TEST_PATH, "test_clean.csv")
    sample_submission = load_csv(SAMPLE_SUBMISSION_PATH, "sample_submission_clean.csv")

    print_section("Phase 2.5: Prepare Features")
    print(f"Loaded clean train: {train.shape[0]:,} rows x {train.shape[1]:,} columns")
    print(f"Loaded clean test: {test.shape[0]:,} rows x {test.shape[1]:,} columns")

    shared_features = get_shared_feature_columns(train, test)
    numeric_features, categorical_features = split_feature_types(train, shared_features)

    outlier_caps = get_outlier_caps(train, numeric_features)

    if CAP_OUTLIERS:
        train, test = apply_outlier_caps(train, test, outlier_caps)

    train_features, test_features = build_feature_tables(
        train,
        test,
        numeric_features,
        categorical_features,
    )

    save_feature_files(
        train,
        test,
        sample_submission,
        train_features,
        test_features,
    )

    report_outputs(
        numeric_features,
        categorical_features,
        train_features,
        test_features,
        outlier_caps,
    )

    print_section("Phase 2.5 Conclusion")
    print("Feature files are ready for Phase 3.")
    print("Phase 3 will split train/validation, scale if needed, train a model, and submit.")


if __name__ == "__main__":
    main()
