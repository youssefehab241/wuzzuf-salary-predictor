import pandas as pd
from pathlib import Path

input_path = Path("data_sources/merged/final_merged_training_standardized.csv")
output_path = Path("data_sources/merged/final_model_training_data.csv")

df = pd.read_csv(input_path)

final_df = pd.DataFrame()
final_df["job_title"] = df["job_title_standardized"].astype(str).str.strip().str.lower()
final_df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
final_df["salary_mid"] = pd.to_numeric(df["salary_mid"], errors="coerce")
final_df["source_name"] = df["source_name"]

print("Before final cleaning:")
print(final_df.head(10))

final_df = final_df.dropna(subset=["job_title", "experience_years", "salary_mid"])
final_df = final_df[final_df["job_title"] != ""]
final_df = final_df[final_df["salary_mid"] >= 1000]
final_df = final_df.drop_duplicates()

final_df.to_csv(output_path, index=False)

print("\nFinal model training file created successfully!")
print(f"Output file: {output_path}")
print(f"Shape: {final_df.shape}")

print("\nCategory counts:")
print(final_df["job_title"].value_counts())

print("\nPreview:")
print(final_df.head(20))