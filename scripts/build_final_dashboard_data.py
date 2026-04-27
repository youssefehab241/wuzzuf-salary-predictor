import pandas as pd
from pathlib import Path


input_path = Path("data_sources/merged/ml_ready_salary_data.csv")
output_path = Path("data_sources/merged/final_dashboard_ready_data.csv")

df = pd.read_csv(input_path)


# ==========================================
# 1. Standardize Job Titles
# ==========================================
def standardize_title(title):
    if pd.isna(title):
        return "other"

    text = str(title).strip().lower()

    # software testing
    if any(word in text for word in [
        "qa", "quality assurance", "testing", "tester", "test engineer",
        "selenium", "playwright", "cypress", "postman", "automation test",
        "manual test", "automation engineer", "software testing", "sdet"
    ]):
        return "software testing engineer"

    # devops / cloud / infra
    if any(word in text for word in [
        "devops", "cloud", "infrastructure", "kafka", "sre",
        "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "ci/cd", "aws", "azure", "gcp",
        "linux admin", "system admin", "site reliability",
        "devsecops", "platform engineer", "cloud engineer"
    ]):
        return "devops engineer"

    # cybersecurity
    if any(word in text for word in [
        "security", "cyber", "soc", "siem", "penetration",
        "pentest", "blue team", "red team", "dfir",
        "information security", "network security",
        "security engineer", "cybersecurity"
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
        "product designer", "figma", "wireframe", "prototype",
        "ui ", " ux ", "web designer"
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
        "pandas", "numpy", "genai", "llm", "mlops", "prompt engineer"
    ]):
        return "data/ai engineer"

    # front-end
    if any(word in text for word in [
        "front end", "frontend", "front-end",
        "react", "angular", "vue", "next.js", "nextjs",
        "javascript", "typescript", "html", "css", "bootstrap",
        "web ui", "frontend developer", "frontend engineer"
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
        "software consultant", "integration engineer", "middleware",
        "developer .net", "developer.net"
    ]):
        return "back end engineer"

    return "other"


df["job_title_standardized"] = df["job_title_clean"].apply(standardize_title)

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

df = df[df["job_title_standardized"].isin(allowed_categories)].copy()


# ==========================================
# 2. Build Location Group
# ==========================================
egypt_keywords = [
    "egypt", "cairo", "giza", "alexandria", "6 october", "6th of october",
    "new cairo", "maadi", "heliopolis", "nasr city", "mansoura", "tanta",
    "suez", "ismailia", "zagazig", "sharm", "hurghada", "asyut", "aswan",
    "minya", "sohag", "qena", "fayoum", "banha", "damanhur", "port said",
    "القاهرة", "الجيزة", "الاسكندرية", "الإسكندرية", "السادس من أكتوبر",
    "اكتوبر", "مصر"
]


def map_location_group(location):
    if pd.isna(location):
        return "Egypt"

    text = str(location).strip().lower()

    if text in ["unknown", "", "nan"]:
        return "Egypt"

    if any(keyword in text for keyword in egypt_keywords):
        return "Egypt"

    return "Outside Egypt"


df["location_group"] = df["location_clean"].apply(map_location_group)


# ==========================================
# 3. Fix Work Mode
# ==========================================
def clean_work_mode(work_mode, location_group):
    if pd.isna(work_mode):
        work_mode = "Unknown"

    text = str(work_mode).strip().lower()

    if text in ["remote"]:
        return "Remote"
    if text in ["hybrid"]:
        return "Hybrid"
    if text in ["on-site", "onsite", "on site"]:
        return "On-Site"

    # Unknown fallback rule
    if location_group == "Outside Egypt":
        return "Remote"
    return "On-Site"


df["work_mode_final"] = df.apply(
    lambda row: clean_work_mode(row["work_mode_clean"], row["location_group"]),
    axis=1
)


# ==========================================
# 4. Fix Seniority from Experience if Unknown
# ==========================================
def infer_seniority(level, exp):
    if pd.notna(level):
        text = str(level).strip().lower()

        if text in ["entry-level", "entry level"]:
            return "Entry-Level"
        if text in ["junior", "jr", "junior level"]:
            return "Junior"
        if text in ["mid-level", "mid level", "mid", "mid career"]:
            return "Mid-Level"
        if text in ["senior", "sr", "senior level"]:
            return "Senior"
        if text in ["lead/manager", "lead", "manager", "lead manager"]:
            return "Lead/Manager"

    if pd.isna(exp):
        return "Entry-Level"

    if exp < 2:
        return "Entry-Level"
    elif exp < 5:
        return "Junior"
    elif exp < 8:
        return "Mid-Level"
    elif exp < 12:
        return "Senior"
    else:
        return "Lead/Manager"


df["level_final"] = df.apply(
    lambda row: infer_seniority(row["level_clean"], row["experience_years_clean"]),
    axis=1
)


# ==========================================
# 5. Salary Cleaning
# ==========================================
# Fix salaries written like 12.5 -> 12500
df.loc[df["salary_target"] < 200, "salary_target"] *= 1000

# Remove unrealistic salaries
df = df[
    (df["salary_target"] >= 3000) &
    (df["salary_target"] <= 250000)
].copy()


# ==========================================
# 6. Final Column Selection
# ==========================================
final_df = df[
    [
        "job_title_standardized",
        "experience_years_clean",
        "location_group",
        "work_mode_final",
        "level_final",
        "salary_target",
    ]
].copy()

final_df = final_df.rename(columns={
    "job_title_standardized": "job_title_clean",
    "work_mode_final": "work_mode_clean",
    "level_final": "level_clean"
})


# ==========================================
# 7. Reporting
# ==========================================
print("Final dataset shape:")
print(final_df.shape)

print("\nJob title counts:")
print(final_df["job_title_clean"].value_counts())

print("\nLocation group counts:")
print(final_df["location_group"].value_counts())

print("\nWork mode counts:")
print(final_df["work_mode_clean"].value_counts())

print("\nSeniority counts:")
print(final_df["level_clean"].value_counts())

print("\nSalary summary:")
print(final_df["salary_target"].describe())

print("\nPreview:")
print(final_df.head(20))


# ==========================================
# 8. Save
# ==========================================
final_df.to_csv(output_path, index=False)

print("\nFinal dashboard-ready file created successfully!")
print(f"Output file: {output_path}")