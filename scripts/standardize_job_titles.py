import pandas as pd
from pathlib import Path


input_path = Path("data_sources/merged/final_merged_training.csv")
output_path = Path("data_sources/merged/final_merged_training_standardized.csv")

df = pd.read_csv(input_path)


def standardize_title(title):
    if pd.isna(title):
        return "other"

    text = str(title).strip().lower()

    # software testing
    if any(word in text for word in [
        "qa", "quality assurance", "testing", "tester", "test engineer",
        "selenium", "playwright", "cypress", "postman", "automation test",
        "manual test", "automation engineer", "software testing"
    ]):
        return "software testing engineer"

    # devops / cloud / infra
    if any(word in text for word in [
        "devops", "cloud", "infrastructure", "kafka", "sre",
        "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "ci/cd", "aws", "azure", "gcp",
        "linux admin", "system admin", "site reliability"
    ]):
        return "devops engineer"

    # cybersecurity
    if any(word in text for word in [
        "security", "cyber", "soc", "siem", "penetration",
        "pentest", "blue team", "red team", "dfir",
        "information security", "network security"
    ]):
        return "cybersecurity engineer"

    # mobile
    if any(word in text for word in [
        "android", "ios", "flutter", "mobile",
        "react native", "swift", "kotlin", "xamarin"
    ]):
        return "mobile engineer"

    # ui/ux
    if any(word in text for word in [
        "ui/ux", "ux", "ui designer", "ux designer",
        "product designer", "figma", "wireframe", "prototype"
    ]):
        return "ui/ux designer"

    # embedded
    if any(word in text for word in [
        "embedded", "firmware", "microcontroller", "microcontrollers",
        "avr", "arm", "stm32", "electronics engineer", "embedded systems"
    ]):
        return "embedded engineer"

    # technical support
    if any(word in text for word in [
        "technical support", "tech support", "it support",
        "support engineer", "help desk", "service desk", "desktop support"
    ]):
        return "technical support engineer"

    # data / ai
    if any(word in text for word in [
        "data scientist", "data science", "data analyst", "data engineer",
        "machine learning", "ml ", " ml", "ai", "artificial intelligence",
        "computer vision", "nlp", "deep learning", "analytics",
        "business intelligence", "bi developer", "bi analyst",
        "power bi", "tableau", "etl", "big data", "spark",
        "pandas", "numpy"
    ]):
        return "data/ai engineer"

    # front-end
    if any(word in text for word in [
        "front end", "frontend", "front-end",
        "react", "angular", "vue", "next.js", "nextjs",
        "javascript", "typescript", "html", "css", "bootstrap",
        "web designer", "web ui"
    ]):
        return "front end engineer"

    # full-stack
    if any(word in text for word in [
        "full stack", "fullstack", "full-stack",
        "mern", "mean", "lamp", "jamstack"
    ]):
        return "full stack engineer"

    # back-end
    if any(word in text for word in [
        "back end", "backend", "back-end",
        ".net", "dotnet", "asp.net", "asp net", "c#", "java", "spring",
        "laravel", "php", "django", "flask", "fastapi",
        "node", "nodejs", "express", "ruby", "rails", "go", "golang",
        "software engineer", "software developer", "backend developer",
        "backend engineer", "api developer", "web developer",
        "oracle", "pl/sql", "plsql", "sql developer",
        "database developer", "db developer", "dba",
        "erp", "odoo", "sap", "crm developer", "technical consultant",
        "software consultant", "integration engineer", "middleware"
    ]):
        return "back end engineer"

    return "other"


df["job_title_standardized"] = df["job_title"].apply(standardize_title)

allowed_categories = [
    "front end engineer",
    "back end engineer",
    "full stack engineer",
    "software testing engineer",
    "devops engineer",
    "data/ai engineer",
    "mobile engineer",
    "cybersecurity engineer",
    "ui/ux designer",
    "embedded engineer",
    "technical support engineer",
]

print("Category counts before filtering:")
print(df["job_title_standardized"].value_counts())

print("\nTop remaining OTHER titles:")
other_counts = (
    df[df["job_title_standardized"] == "other"]["job_title"]
    .value_counts()
    .head(100)
)
print(other_counts)

filtered_df = df[df["job_title_standardized"].isin(allowed_categories)].copy()

print("\nCategory counts after filtering:")
print(filtered_df["job_title_standardized"].value_counts())

filtered_df.to_csv(output_path, index=False)

print("\nStandardized file created successfully!")
print(f"Output file: {output_path}")
print(f"Shape: {filtered_df.shape}")

print("\nPreview:")
print(filtered_df[["job_title", "job_title_standardized", "experience_years", "salary_mid"]].head(20))