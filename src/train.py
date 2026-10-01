import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/california_housing.csv"
)

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "MedHouseVal"


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "House Price Prediction"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HOUSE PRICE PREDICTION - MLOPS TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print(
    f"\nFeatures: {X.shape[1]}"
)

print(
    f"Target: {TARGET}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": 100,
            "random_state": 42
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        }
    }
}


# ============================================================
# RESULTS
# ============================================================

results = []

trained_models = {}


# ============================================================
# TRAIN MODELS WITH MLFLOW
# ============================================================

for model_name, model_info in models.items():

    model = model_info["model"]
    params = model_info["params"]

    print("\n" + "=" * 70)

    print(
        f"Training: {model_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Create preprocessing + model pipeline
        # ----------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    model
                )
            ]
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = pipeline.predict(
            X_test
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        if params:

            mlflow.log_params(
                params
            )


        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "r2",
            r2
        )


        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
    pipeline,
    name="model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
)


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
        )

        trained_models[
            model_name
        ] = pipeline


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"MAE : {mae:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"R²  : {r2:.4f}"
        )

        print(
            "MLflow run completed."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE"
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[
    0
]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)

print(
    f"BEST MODEL: {best_model_name}"
)

print("=" * 70)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "house_price_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print(
    f"\nBest model saved to:"
)

print(
    model_path
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_path = (
    MODEL_DIR /
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nModel comparison saved to:"
)

print(
    results_path
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "MLFLOW EXPERIMENT TRACKING COMPLETED"
)

print("=" * 70)

print(
    "\nMLflow tracking directory:"
)

print(
    "mlruns/"
)