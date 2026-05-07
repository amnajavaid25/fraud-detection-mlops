import pandas as pd

df = pd.read_csv("data/train_transaction.csv")

print("Dataset shape:", df.shape)

# schema checks
assert "isFraud" in df.columns, "Target column missing"

# missing value checks
missing_ratio = df.isnull().mean().mean()

print("Missing ratio:", missing_ratio)

if missing_ratio > 0.5:
    raise ValueError("Too many missing values")

print("Validation successful")