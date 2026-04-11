import pandas as pd
from pathlib import Path

file_path = Path("data_sources/training/wuzzuf_cleaned_training_data.csv")

df = pd.read_csv(file_path)

print("File loaded successfully!")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())