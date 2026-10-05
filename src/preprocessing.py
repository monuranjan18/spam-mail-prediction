from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

RAW_DATA_PATH = Path("data/mail_data.csv")
CLEANED_DATA_PATH = Path("data/cleaned_mail_data.csv")


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the raw spam mail dataset.

    UTF-8 is attempted first. If decoding fails,
    latin1 is used as a fallback.
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
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="latin1"
        )

    return df


# ============================================================
# Validate Dataset
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the structure and basic quality of the dataset.
    """

    required_columns = {"Category", "Message"}

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Check missing values
    if df["Category"].isna().any():
        raise ValueError(
            "Category column contains missing values."
        )

    if df["Message"].isna().any():
        raise ValueError(
            "Message column contains missing values."
        )

    # Check labels
    valid_labels = {"ham", "spam"}

    actual_labels = set(
        df["Category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    invalid_labels = actual_labels - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels found: "
            f"{sorted(invalid_labels)}"
        )


# ============================================================
# Clean Dataset
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset without performing aggressive NLP
    preprocessing.
    """

    cleaned_df = df.copy()

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    cleaned_df = cleaned_df[
        ["Category", "Message"]
    ]

    # --------------------------------------------------------
    # Normalize category labels
    # --------------------------------------------------------

    cleaned_df["Category"] = (
        cleaned_df["Category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Remove surrounding whitespace from messages
    #
    # We are NOT removing punctuation, numbers,
    # URLs, currency symbols, etc. at this stage.
    # --------------------------------------------------------

    cleaned_df["Message"] = (
        cleaned_df["Message"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove empty messages
    # --------------------------------------------------------

    before_empty_removal = len(cleaned_df)

    cleaned_df = cleaned_df[
        cleaned_df["Message"].str.len() > 0
    ].copy()

    empty_removed = (
        before_empty_removal - len(cleaned_df)
    )

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    before_duplicates = len(cleaned_df)

    cleaned_df = (
        cleaned_df
        .drop_duplicates(
            subset=["Category", "Message"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_duplicates - len(cleaned_df)
    )

    return cleaned_df, empty_removed, duplicates_removed


# ============================================================
# Print Cleaning Report
# ============================================================

def print_cleaning_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    empty_removed: int,
    duplicates_removed: int,
) -> None:
    """
    Display a summary of the cleaning process.
    """

    print("\n" + "=" * 70)
    print("DATA CLEANING REPORT")
    print("=" * 70)

    print("\nOriginal dataset:")
    print(f"Rows: {len(original_df)}")

    print("\nRows removed because of empty messages:")
    print(empty_removed)

    print("\nDuplicate rows removed:")
    print(duplicates_removed)

    print("\nCleaned dataset:")
    print(f"Rows: {len(cleaned_df)}")

    print("\nRows removed in total:")
    print(
        len(original_df) - len(cleaned_df)
    )

    print("\nCleaned class distribution:")

    print(
        cleaned_df["Category"]
        .value_counts()
    )

    print("\nCleaned class distribution (%):")

    print(
        (
            cleaned_df["Category"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print("\nRemaining duplicate rows:")

    print(
        cleaned_df.duplicated().sum()
    )

    print("\nMissing values after cleaning:")

    print(
        cleaned_df.isnull().sum()
    )

    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Execute the dataset cleaning pipeline.
    """

    print("Loading raw dataset...")

    df = load_dataset(RAW_DATA_PATH)

    print("Validating dataset...")

    validate_dataset(df)

    print("Cleaning dataset...")

    cleaned_df, empty_removed, duplicates_removed = (
        clean_dataset(df)
    )

    print_cleaning_report(
        original_df=df,
        cleaned_df=cleaned_df,
        empty_removed=empty_removed,
        duplicates_removed=duplicates_removed,
    )

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------

    cleaned_df.to_csv(
        CLEANED_DATA_PATH,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nCleaned dataset saved to:"
        f"\n{CLEANED_DATA_PATH}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()