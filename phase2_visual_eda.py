"""
Phase 2: Visual EDA

Use the cleaned training data to make simple charts and feature summaries.

Goal:
Get clues about the target, features, outliers, correlations, and mutual
information before building the first model.

This script does not train a model.
It saves plots and summaries into reports/phase2_visual_eda/.
"""
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is not installed. Run `uv sync` or `uv add matplotlib`, "
        "then run this script again."
    ) from exc

try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is not installed. Run `uv sync`, then run this script again."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_clean.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase2_visual_eda"

ID_COLUMN = "Id"
DATE_COLUMN = "Date"
TARGET_RETURN = "Target_Return"
TARGET_DIRECTION = "Target_Direction"
TARGET_COLUMNS = [TARGET_RETURN, TARGET_DIRECTION]

KEY_FEATURES = [
    "Return_1d",
    "Return_5d",
    "RSI_14",
    "MACD",
    "Volatility_20d",
    "Volume",
]


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


def get_numeric_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    features = []

    for column in train.columns:
        if column in TARGET_COLUMNS or column in [ID_COLUMN, DATE_COLUMN]:
            continue

        if column not in test.columns:
            continue

        if pd.api.types.is_numeric_dtype(train[column]):
            features.append(column)

    return features


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_target_return_histogram(train: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(train[TARGET_RETURN], bins=60, edgecolor="black")
    plt.title("Target_Return Distribution")
    plt.xlabel("Target_Return")
    plt.ylabel("Row count")
    save_plot(OUTPUT_DIR / "01_target_return_distribution.png")


def plot_target_direction_counts(train: pd.DataFrame) -> None:
    counts = train[TARGET_DIRECTION].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Target_Direction Class Balance")
    plt.xlabel("Target_Direction")
    plt.ylabel("Row count")
    save_plot(OUTPUT_DIR / "02_target_direction_counts.png")


def plot_returns_over_id(train: pd.DataFrame) -> None:
    rolling_average = train[TARGET_RETURN].rolling(window=100, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(
        train[ID_COLUMN],
        train[TARGET_RETURN],
        linewidth=0.5,
        alpha=0.35,
        label="Daily return",
    )
    plt.plot(train[ID_COLUMN], rolling_average, linewidth=2, label="100-row rolling average")
    plt.title("Target_Return Over ID Order")
    plt.xlabel("Id")
    plt.ylabel("Target_Return")
    plt.legend()
    save_plot(OUTPUT_DIR / "03_target_return_over_id.png")


def plot_feature_scatter_grid(train: pd.DataFrame, numeric_features: list[str]) -> None:
    features_to_plot = [feature for feature in KEY_FEATURES if feature in numeric_features]

    if not features_to_plot:
        print("No key features found for scatter plots.")
        return

    sample_size = min(1200, len(train))
    plot_data = train.sample(n=sample_size, random_state=42)

    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7))
    axes = axes.flatten()

    for index, feature in enumerate(features_to_plot):
        axes[index].scatter(plot_data[feature], plot_data[TARGET_RETURN], s=8, alpha=0.35)
        axes[index].set_title(feature)
        axes[index].set_xlabel(feature)
        axes[index].set_ylabel(TARGET_RETURN)

    for index in range(len(features_to_plot), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Feature vs Target_Return", fontsize=14)
    save_plot(OUTPUT_DIR / "04_feature_scatter_grid.png")


def get_target_correlations(train: pd.DataFrame, numeric_features: list[str]) -> pd.Series:
    columns = numeric_features + [TARGET_RETURN]
    correlations = train[columns].corr()[TARGET_RETURN]
    correlations = correlations.drop(TARGET_RETURN)
    return correlations.sort_values(key=lambda values: values.abs(), ascending=False)


def plot_correlation_heatmap(train: pd.DataFrame, correlations: pd.Series) -> None:
    top_features = correlations.head(8).index.tolist()
    columns = top_features + [TARGET_RETURN]
    corr_matrix = train[columns].corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
    plt.yticks(range(len(columns)), columns)
    plt.title("Correlation Heatmap: Top Features and Target_Return")
    save_plot(OUTPUT_DIR / "05_correlation_heatmap.png")


def save_correlation_summary(correlations: pd.Series) -> None:
    summary = correlations.reset_index()
    summary.columns = ["feature", "correlation_with_Target_Return"]

    output_path = OUTPUT_DIR / "correlation_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def get_mutual_information(train: pd.DataFrame, numeric_features: list[str]) -> pd.Series:
    X = train[numeric_features]
    y = train[TARGET_RETURN]

    scores = mutual_info_regression(X, y, random_state=42)
    scores = pd.Series(scores, index=numeric_features)
    return scores.sort_values(ascending=False)


def save_mutual_information_summary(mi_scores: pd.Series) -> None:
    summary = mi_scores.reset_index()
    summary.columns = ["feature", "mutual_information_with_Target_Return"]

    output_path = OUTPUT_DIR / "mutual_information_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def get_outlier_summary(train: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows = []

    for feature in numeric_features:
        values = train[feature]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        outlier_count = ((values < lower_limit) | (values > upper_limit)).sum()

        rows.append(
            {
                "feature": feature,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_count / len(train),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("outlier_percent", ascending=False)


def save_outlier_summary(outliers: pd.DataFrame) -> None:
    output_path = OUTPUT_DIR / "outlier_summary.csv"
    outliers.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def report_correlation_clues(correlations: pd.Series) -> None:
    print_section("Correlation Clues")
    print("Features most related to Target_Return by absolute correlation:")

    for feature, value in correlations.head(10).items():
        print(f"- {feature}: {value:.4f}")

    print("\nCorrelation is only a clue, not proof a feature will help the model.")


def report_mutual_information_clues(mi_scores: pd.Series) -> None:
    print_section("Mutual Information Clues")
    print("Features with highest mutual information with Target_Return:")

    for feature, value in mi_scores.head(10).items():
        print(f"- {feature}: {value:.4f}")

    print("\nMutual information can catch non-linear relationships that correlation may miss.")


def report_outlier_clues(outliers: pd.DataFrame) -> None:
    print_section("Outlier Clues")
    print("Features with the highest outlier percentage by the IQR rule:")

    for row in outliers.head(10).itertuples(index=False):
        print(f"- {row.feature}: {row.outlier_percent:.2%} outliers")

    print("\nOutliers are not automatically bad. For stock data, big moves may be real signal.")


def main() -> None:
    train = load_csv(TRAIN_PATH, "train_clean.csv")
    test = load_csv(TEST_PATH, "test_clean.csv")
    numeric_features = get_numeric_features(train, test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Phase 2: Visual EDA")
    print(f"Loaded clean train: {train.shape[0]:,} rows x {train.shape[1]:,} columns")
    print(f"Loaded clean test: {test.shape[0]:,} rows x {test.shape[1]:,} columns")
    print(f"Numeric features available: {len(numeric_features)}")
    print(f"Output folder: {OUTPUT_DIR}")

    print_section("Creating Plots")
    plot_target_return_histogram(train)
    plot_target_direction_counts(train)
    plot_returns_over_id(train)
    plot_feature_scatter_grid(train, numeric_features)

    correlations = get_target_correlations(train, numeric_features)
    plot_correlation_heatmap(train, correlations)
    save_correlation_summary(correlations)

    mi_scores = get_mutual_information(train, numeric_features)
    save_mutual_information_summary(mi_scores)

    outliers = get_outlier_summary(train, numeric_features)
    save_outlier_summary(outliers)

    report_correlation_clues(correlations)
    report_mutual_information_clues(mi_scores)
    report_outlier_clues(outliers)

    print_section("Phase 2 Conclusion")
    print("Visual EDA is done.")
    print("Next phase: prepare feature files for modeling.")


if __name__ == "__main__":
    main()
