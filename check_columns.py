from pathlib import Path

import pandas as pd

# Path to training data
path = Path("data/processed/train_test/X_train.csv").resolve()

print("=" * 80)
print("FILE BEING READ:")
print(path)
print("=" * 80)

# Load data
X = pd.read_csv(path)

print("\nShape:")
print(X.shape)

print("\nColumns:")
print(X.columns.tolist())

print("\nData Types:")
print(X.dtypes)

print("\nObject (string) columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())

print("\nString dtype columns:")
print(X.select_dtypes(include=["string"]).columns.tolist())

print("\nCategorical columns:")
print(X.select_dtypes(include=["category"]).columns.tolist())

print("\nNon-numeric columns:")
non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
print(non_numeric)

print("\nFirst 5 rows:")
print(X.head())

print("=" * 80)
