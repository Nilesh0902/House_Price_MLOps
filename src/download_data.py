from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path


# ============================================================
# LOAD DATASET
# ============================================================

print("Downloading California Housing dataset...")

housing = fetch_california_housing(
    as_frame=True
)

df = housing.frame


# ============================================================
# DISPLAY BASIC INFORMATION
# ============================================================

print("\nDataset downloaded successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# CREATE DATA DIRECTORY
# ============================================================

output_dir = Path("data/raw")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE DATASET
# ============================================================

output_file = output_dir / "california_housing.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nDataset saved successfully!")
print(f"Location: {output_file}")