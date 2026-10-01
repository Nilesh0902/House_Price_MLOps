import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/california_housing.csv")

OUTPUT_DIR = Path("data/eda")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("HOUSE PRICE DATASET - EDA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("1. DATASET SHAPE")
print("=" * 60)

print(df.shape)


print("\n" + "=" * 60)
print("2. COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())


print("\n" + "=" * 60)
print("3. DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print(
    f"\nTotal missing values: "
    f"{missing_values.sum()}"
)


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("5. DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(
    f"Duplicate rows: {duplicates}"
)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("6. DESCRIPTIVE STATISTICS")
print("=" * 60)

print(
    df.describe().round(3)
)


# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET = "MedHouseVal"

print("\n" + "=" * 60)
print("7. TARGET VARIABLE")
print("=" * 60)

print(f"Target: {TARGET}")

print(
    f"Minimum: {df[TARGET].min():.3f}"
)

print(
    f"Maximum: {df[TARGET].max():.3f}"
)

print(
    f"Mean: {df[TARGET].mean():.3f}"
)

print(
    f"Median: {df[TARGET].median():.3f}"
)


# ============================================================
# CORRELATION
# ============================================================

print("\n" + "=" * 60)
print("8. CORRELATION WITH TARGET")
print("=" * 60)

correlation = (
    df.corr(numeric_only=True)[TARGET]
    .sort_values(ascending=False)
)

print(
    correlation.round(3)
)


# ============================================================
# SAVE CORRELATION
# ============================================================

correlation.to_csv(
    OUTPUT_DIR / "target_correlations.csv"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df[TARGET],
    bins=40,
    kde=True
)

plt.title(
    "Distribution of Median House Value"
)

plt.xlabel(
    "Median House Value"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "target_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=150
)

plt.close()


# ============================================================
# FEATURE VS TARGET PLOTS
# ============================================================

features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]


for feature in features:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[feature],
        df[TARGET],
        alpha=0.3
    )

    plt.title(
        f"{feature} vs {TARGET}"
    )

    plt.xlabel(feature)

    plt.ylabel(TARGET)

    plt.tight_layout()

    safe_name = feature.lower()

    plt.savefig(
        OUTPUT_DIR / f"{safe_name}_vs_target.png",
        dpi=150
    )

    plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nEDA files saved in: {OUTPUT_DIR}"
)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):

    print(
        f" - {file.name}"
    )