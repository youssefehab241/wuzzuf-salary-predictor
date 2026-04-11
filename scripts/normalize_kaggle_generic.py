import pandas as pd
from pathlib import Path
import numpy as np
import re


# =========================
# DATASET CONFIG
# =========================
DATASET_NAME = "developers_2025"
INPUT_FILE = "data_sources/raw/teammates/developers_2025/developers_2025.csv"

EXP_COL = "Years of Experience:"
LEVEL_COL = "Job Title ( Position )"
CURRENCY_COL = "Currency of Salary:"
SALARY_COL = "Monthly Salary:"
DATE_COL = "طابع زمني"
MAIN_TECH_COL = "Main Tech:"

FILE_TYPE = "csv"
SHEET_NAME = 0

# =========================
# PROJECT FX ASSUMPTIONS
# =========================
USD_TO_EGP = 50
EUR_TO_EGP = 55
SAR_TO_EGP = 13
AED_TO_EGP = 13.6
PKR_TO_EGP = 0.18
JOD_TO_EGP = 70


def parse_experience(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if text == "":
        return np.nan

    if "less than" in text and "1" in text:
        return 0.5

    if "to" in text or "-" in text:
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            low = float(nums[0])
            high = float(nums[1])
            return round((low + high) / 2, 2)

    if "month" in text:
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if nums:
            months = float(nums[0])
            return round(months / 12, 2)

    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[0])

    return np.nan


def normalize_currency(currency_value):
    if pd.isna(currency_value):
        return "EGP"

    text = str(currency_value).strip().upper()

    if "USD" in text or "$" in text:
        return "USD"
    if "EGP" in text:
        return "EGP"
    if "SAR" in text or "SR" in text:
        return "SAR"
    if "EUR" in text:
        return "EUR"
    if "AED" in text:
        return "AED"
    if "PKR" in text:
        return "PKR"
    if "JD" in text or "JOD" in text:
        return "JOD"

    return text


def parse_salary_number(raw_value):
    if pd.isna(raw_value):
        return np.nan

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return float(raw_value)

    text = str(raw_value).strip().lower()
    if text == "":
        return np.nan

    text = text.replace(",", "")
    text = text.replace("egp", "")
    text = text.replace("usd", "")
    text = text.replace("eur", "")
    text = text.replace("euro", "")
    text = text.replace("sar", "")
    text = text.replace("aed", "")
    text = text.replace("pkr", "")
    text = text.replace("jod", "")
    text = text.replace("jd", "")
    text = text.replace("$", "")
    text = text.replace("جنيه", "")
    text = text.replace("ج.م", "")
    text = text.strip()

    has_k = "k" in text

    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan

    values = [float(num) for num in nums]

    if has_k:
        values = [value * 1000 for value in values]

    if len(values) >= 2 and ("to" in text or "-" in text):
        return round((values[0] + values[1]) / 2, 2)

    return values[0]


def convert_to_egp(amount, currency):
    if pd.isna(amount):
        return np.nan

    if currency == "USD":
        return amount * USD_TO_EGP
    if currency == "EUR":
        return amount * EUR_TO_EGP
    if currency == "SAR":
        return amount * SAR_TO_EGP
    if currency == "AED":
        return amount * AED_TO_EGP
    if currency == "PKR":
        return amount * PKR_TO_EGP
    if currency == "JOD":
        return amount * JOD_TO_EGP

    return amount


def map_main_tech_to_title(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if "front end" in text:
        return "front end engineer"
    if "back end" in text:
        return "back end engineer"
    if "full stack" in text:
        return "full stack engineer"
    if "mobile" in text:
        return "mobile engineer"
    if "devops" in text:
        return "devops engineer"
    if "data" in text:
        return "data engineer"
    if "ai" in text or "machine learning" in text:
        return "ai engineer"
    if "cyber" in text or "security" in text:
        return "cybersecurity engineer"
    if "qa" in text or "testing" in text:
        return "software testing engineer"
    if "ui" in text or "ux" in text:
        return "ui/ux designer"

    return text


input_path = Path(INPUT_FILE)
processed_output_path = Path(f"data_sources/processed/{DATASET_NAME}_normalized.csv")
training_output_path = Path(f"data_sources/training/{DATASET_NAME}_training.csv")

if FILE_TYPE == "excel":
    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)
elif FILE_TYPE == "csv":
    df = pd.read_csv(input_path)
else:
    raise ValueError("FILE_TYPE must be either 'excel' or 'csv'")

df.columns = df.columns.str.strip()

normalized = pd.DataFrame(index=df.index)
normalized["job_title"] = df[MAIN_TECH_COL].apply(map_main_tech_to_title)
normalized["experience_years"] = df[EXP_COL].apply(parse_experience)
normalized["position_level_raw"] = df[LEVEL_COL].astype(str).str.strip().str.lower()
normalized["main_tech_raw"] = df[MAIN_TECH_COL].astype(str).str.strip()

normalized["salary_original"] = df[SALARY_COL]
normalized["currency_original"] = df[CURRENCY_COL]
normalized["currency_detected"] = df[CURRENCY_COL].apply(normalize_currency)
normalized["salary_original_numeric"] = df[SALARY_COL].apply(parse_salary_number)

normalized["salary_mid"] = normalized.apply(
    lambda row: convert_to_egp(row["salary_original_numeric"], row["currency_detected"]),
    axis=1
)

if DATE_COL in df.columns:
    normalized["salary_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
else:
    normalized["salary_date"] = pd.NaT

normalized["source_name"] = f"Teammate_{DATASET_NAME}"

print("Before dropna:")
print(
    normalized[
        [
            "job_title",
            "experience_years",
            "position_level_raw",
            "main_tech_raw",
            "salary_original",
            "currency_original",
            "currency_detected",
            "salary_mid",
        ]
    ].head(10)
)

print("\nJob title counts:")
print(normalized["job_title"].value_counts(dropna=False).head(20))

print("\nCurrency counts:")
print(normalized["currency_detected"].value_counts(dropna=False))

normalized = normalized.dropna(subset=["job_title", "experience_years", "salary_mid"])
normalized = normalized[normalized["job_title"] != ""]
normalized = normalized.drop_duplicates()

training_df = normalized[["job_title", "experience_years", "salary_mid", "source_name"]].copy()

processed_output_path.parent.mkdir(parents=True, exist_ok=True)
training_output_path.parent.mkdir(parents=True, exist_ok=True)

normalized.to_csv(processed_output_path, index=False)
training_df.to_csv(training_output_path, index=False)

print("\nFiles created successfully!")
print(f"Processed file: {processed_output_path}")
print(f"Training file: {training_output_path}")
print(f"Processed shape: {normalized.shape}")
print(f"Training shape: {training_df.shape}")

print("\nTraining preview:")
print(training_df.head(10))