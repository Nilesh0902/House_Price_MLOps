import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/california_housing.csv"
)

PROCESSED_DIR = Path(
    "data/processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):

    target_column = "MedHouseVal"

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor():

    preprocessor = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    return preprocessor


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HOUSE PRICE DATA PREPROCESSING")
    print("=" * 60)

    # Load
    df = load_data()

    print(
        f"\nDataset shape: {df.shape}"
    )

    # Features and target
    X, y = prepare_data(df)

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Target: MedHouseVal"
    )

    # Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data
    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "\nPreprocessing completed successfully."
    )

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    print(
        "\nScaler fitted only on training data."
    )

    print(
        "No data leakage introduced."
    )