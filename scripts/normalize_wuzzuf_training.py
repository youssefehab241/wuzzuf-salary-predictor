import pandas as pd
from pathlib import Path
import numpy as np


INPUT_FILE = "data_sources/training/wuzzuf_cleaned_training_data.csv"
OUTPUT_FILE = "data_sources/training/wuzzuf_training_standardized.csv"


input_path = Path(INPUT_FILE)
output_path = Path(OUTPUT_FILE)

df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

normalized = pd.DataFrame(index=df.index)

normalized["job_title"] = df["Title"].astype(str).str.strip().str.lower()
normalized["experience_years"] = pd.to_numeric(df["Min_Experience_Years"], errors="coerce")
normalized["salary_mid"] = pd.to_numeric(df["Avg_Salary"], errors="coerce")
normalized["source_name"] = "Wuzzuf"

print("Before cleaning:")
print(normalized.head(10))

normalized = normalized.dropna(subset=["job_title", "experience_years", "salary_mid"])
normalized = normalized[normalized["job_title"] != ""]
normalized = normalized[normalized["salary_mid"] >= 1000]
normalized = normalized.drop_duplicates()

output_path.parent.mkdir(parents=True, exist_ok=True)
normalized.to_csv(output_path, index=False)

print("\nFile created successfully!")
print(f"Output file: {output_path}")
print(f"Shape: {normalized.shape}")

print("\nPreview:")
print(normalized.head(10))