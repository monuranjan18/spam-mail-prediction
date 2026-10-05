from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/mail_data.csv")


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the spam mail dataset.

    The script first attempts UTF-8 encoding.
    If that fails, it falls back to latin1.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8"
        )

        print("Dataset loaded using UTF-8 encoding.")

    except UnicodeDecodeError:
        print(
            "UTF-8 decoding failed. "
            "Loading dataset using latin1 encoding..."
        )

        df = pd.read_csv(
            path,
            encoding="latin1"
        )

        print("Dataset loaded successfully using latin1 encoding.")

    return df


# ============================================================
# Dataset Inspection
# ============================================================

def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Display important information about the dataset.
    """

    print("\n" + "=" * 70)
    print("SPAM MAIL DATASET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Dataset Shape
    # --------------------------------------------------------

    print("\n1. Dataset Shape")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # 2. Column Names
    # --------------------------------------------------------

    print("\n2. Column Names")

    for column in df.columns:
        print(f"- {column}")

    # --------------------------------------------------------
    # 3. Data Types
    # --------------------------------------------------------

    print("\n3. Data Types")

    print(df.dtypes)

    # --------------------------------------------------------
    # 4. Missing Values
    # --------------------------------------------------------

    print("\n4. Missing Values")

    missing_values = df.isnull().sum()

    print(missing_values)

    total_missing = missing_values.sum()

    print(f"\nTotal missing values: {total_missing}")

    # --------------------------------------------------------
    # 5. Duplicate Rows
    # --------------------------------------------------------

    print("\n5. Duplicate Rows")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    # --------------------------------------------------------
    # 6. Unique Values
    # --------------------------------------------------------

    print("\n6. Unique Values Per Column")

    for column in df.columns:

        print(f"\n{column}:")
        print(f"Unique values: {df[column].nunique()}")

    # --------------------------------------------------------
    # 7. Dataset Preview
    # --------------------------------------------------------

    print("\n7. Dataset Preview")

    print(df.head())

    # --------------------------------------------------------
    # 8. Target / Class Distribution
    # --------------------------------------------------------

    print("\n8. Target / Class Distribution")

    category_counts = df["Category"].value_counts()

    print("\nClass counts:")

    print(category_counts)

    # --------------------------------------------------------
    # Class Percentages
    # --------------------------------------------------------

    print("\nClass percentages:")

    category_percentages = (
        df["Category"]
        .value_counts(normalize=True)
        * 100
    )

    print(category_percentages.round(2))

    # --------------------------------------------------------
    # Class Distribution Table
    # --------------------------------------------------------

    print("\nClass distribution table:")

    class_distribution = pd.DataFrame(
        {
            "Count": category_counts,
            "Percentage": category_percentages.round(2)
        }
    )

    print(class_distribution)

    # --------------------------------------------------------
    # 9. Duplicate Analysis
    # --------------------------------------------------------

    print("\n9. Duplicate Analysis")

    print(
        f"Total duplicate rows: {duplicate_count}"
    )

    if duplicate_count > 0:

        duplicate_rows = (
            df[df.duplicated(keep=False)]
            .sort_values(
                by=["Message", "Category"]
            )
        )

        print("\nDuplicate rows by category:")

        print(
            duplicate_rows["Category"]
            .value_counts()
        )

        print("\nExample duplicate records:")

        print(
            duplicate_rows.head(10)
        )

    else:

        print("No duplicate rows found.")

    # --------------------------------------------------------
    # 10. Text Statistics
    # --------------------------------------------------------

    print("\n10. Text / Message Statistics")

    message_lengths = df["Message"].astype(str).str.len()

    print(
        f"Minimum message length : "
        f"{message_lengths.min()} characters"
    )

    print(
        f"Maximum message length : "
        f"{message_lengths.max()} characters"
    )

    print(
        f"Average message length : "
        f"{message_lengths.mean():.2f} characters"
    )

    print(
        f"Median message length  : "
        f"{message_lengths.median():.2f} characters"
    )

    # --------------------------------------------------------
    # 11. Message Statistics by Class
    # --------------------------------------------------------

    print("\n11. Message Length by Class")

    class_message_stats = (
        df.assign(
            message_length=message_lengths
        )
        .groupby("Category")["message_length"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
        .round(2)
    )

    print(class_message_stats)

    # --------------------------------------------------------
    # 12. Sample Spam Messages
    # --------------------------------------------------------

    print("\n12. Sample SPAM Messages")

    spam_messages = df[
        df["Category"].str.lower() == "spam"
    ]["Message"]

    print(
        spam_messages.head(5).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 13. Sample HAM Messages
    # --------------------------------------------------------

    print("\n13. Sample HAM Messages")

    ham_messages = df[
        df["Category"].str.lower() == "ham"
    ]["Message"]

    print(
        ham_messages.head(5).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 14. Dataset Information
    # --------------------------------------------------------

    print("\n14. Dataset Information")

    df.info()

    # --------------------------------------------------------
    # 15. Memory Usage
    # --------------------------------------------------------

    print("\n15. Memory Usage")

    memory_usage_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    print(
        f"Memory usage: "
        f"{memory_usage_mb:.4f} MB"
    )

    # --------------------------------------------------------
    # 16. Basic Dataset Validation
    # --------------------------------------------------------

    print("\n16. Basic Dataset Validation")

    required_columns = {
        "Category",
        "Message"
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:

        print(
            "WARNING: Required columns missing:"
        )

        for column in missing_columns:
            print(f"- {column}")

    else:

        print(
            "Required columns found: "
            "Category, Message"
        )

    unique_categories = (
        df["Category"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    print(
        "\nUnique target labels:"
    )

    for label in unique_categories:
        print(f"- {label}")

    print("\n" + "=" * 70)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Main program entry point.
    """

    try:

        dataset = load_dataset(DATA_PATH)

        inspect_dataset(dataset)

    except Exception as error:

        print("\nERROR:")
        print(error)

        raise


if __name__ == "__main__":
    main()