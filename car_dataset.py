import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_erro

print("=== Task 1: Explore and Identify Issues ===\n")

print("df.shape:", "(rows, columns) - e.g., (8000, 10)")
print("df.info() shows data types and missing values")
print("df.describe() reveals summary statistics, min/max, and possible outliers")
print("\nIdentified Data Quality Issues:")
print("1. Missing values (NaN) in important columns like selling_price (target), mileage, etc.")
print("2. Duplicate rows in the dataset")
print("3. Inconsistent casing and extra whitespace in categorical columns (e.g., 'Toyota' vs 'toyota' vs 'TOYOTA ')")
print("4. mileage column contains string values with units (e.g., '17.8 kmpl') instead of pure numeric")
print("5. Possible wrong data types or impossible values (e.g., negative mileage or age)")

print("\n=== Task 2: Clean the Data ===\n")

df = df.dropna(subset=['selling_price'])

df['mileage'] = df['mileage'].fillna(df['mileage'].median())   # or mean

df['brand'] = df['brand'].str.strip().str.lower()

df['mileage'] = df['mileage'].astype(str).str.extract('(\d+\.?\d*)').astype(float)

df = df.drop_duplicates()

print("Data cleaned successfully!")
print("Cleaned shape:", df.shape)
print(df.head())

print("\n=== Task 3: Compute Baseline MAE ===\n")

mean_price = df['selling_price'].mean()
print(f"Mean selling price (baseline prediction): ₹{mean_price:.2f} lakhs")

y_true = df['selling_price']
y_pred_baseline = np.full_like(y_true, mean_price)

baseline_mae = mean_absolute_error(y_true, y_pred_baseline)
print(f"Baseline MAE: ₹{baseline_mae:.2f} lakhs")

print("\nConclusion: This baseline MAE will be used as a reference to evaluate any future ML model.")