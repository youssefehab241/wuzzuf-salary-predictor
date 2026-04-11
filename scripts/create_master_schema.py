import pandas as pd
from pathlib import Path

columns = [
    "job_title",
    "job_family",
    "experience_years_min",
    "experience_years_max",
    "experience_level",
    "salary_min",
    "salary_max",
    "salary_mid",
    "currency",
    "country",
    "city",
    "work_mode",
    "company_name",
    "source_name",
    "source_type",
    "confidence_score",
    "record_url",
    "date_collected",
]

output_path = Path("data_sources/merged/master_schema_template.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(columns=columns)
df.to_csv(output_path, index=False)

print(f"Master schema template created at: {output_path}")