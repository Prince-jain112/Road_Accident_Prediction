import pandas as pd

# Dataset Load
df = pd.read_csv("dataset.csv")

print("✅ Dataset Successfully Loaded")

# First 5 Rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Shape
print("\n========== SHAPE ==========")
print(df.shape)

# Columns
print("\n========== COLUMNS ==========")
print(df.columns)

# Data Types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Missing Values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Dataset Information
print("\n========== DATASET INFO ==========")
print(df.info())

# Duplicate Rows

print("\nDuplicate Rows :", df.duplicated().sum())

# Missing Values

print("\nMissing Values")
print(df.isnull().sum())

# Festival Column Remove

df = df.drop(columns=["festival"])

print("\nFestival Column Removed")

print(df.columns)

df.to_csv("clean_dataset.csv", index=False)

print("Clean Dataset Saved Successfully")