import pandas as pd
from pathlib import Path

training_dir = Path("data_sources/training")
output_path = Path("data_sources/merged/final_merged_training.csv")

file_names = [
    "developers_2025_training.csv",
    "Developers_Salaries_in_2024_training.csv",
    "egyptian_salaries_2024_training.csv",
    "EgyTech_participants_data_training.csv",
    "Front_End_Developer_salaries_in_Egypt_2023_training.csv",
    "software_testing_egypt_2023_training.csv",
    "Web_Developers_Salaries_Egypt_2024_training.csv",
    "wuzzuf_training_standardized.csv",
]

dfs = []

for file_name in file_names:
    file_path = training_dir / file_name
    if file_path.exists():
        df = pd.read_csv(file_path)
        df["source_file"] = file_name
        dfs.append(df)
        print(f"Loaded: {file_name} -> {df.shape}")
    else:
        print(f"Missing: {file_name}")

merged_df = pd.concat(dfs, ignore_index=True)

print("\nBefore cleaning:")
print(merged_df.shape)

merged_df = merged_df.dropna(subset=["job_title", "experience_years", "salary_mid"])
merged_df = merged_df[merged_df["job_title"].astype(str).str.strip() != ""]
merged_df = merged_df[merged_df["salary_mid"] >= 1000]
merged_df = merged_df.drop_duplicates()

output_path.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_path, index=False)

print("\nFinal merged file created successfully!")
print(f"Output file: {output_path}")
print(f"Final shape: {merged_df.shape}")

print("\nPreview:")
print(merged_df.head(10))