import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Egypt Tech Salary Predictor",
    page_icon="💼",
    layout="centered"
)

MODEL_PATH = Path("salary_prediction_model/optimized_salary_model.pkl")
DATA_PATH = Path("data_sources/merged/ml_ready_salary_data_standardized.csv")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=["job_title_clean"], errors="ignore")
    df = df.rename(columns={"job_title_standardized": "job_title_clean"})

    df.loc[df["salary_target"] < 200, "salary_target"] *= 1000
    df = df[(df["salary_target"] >= 5000) & (df["salary_target"] <= 250000)].copy()

    df = df[
        [
            "job_title_clean",
            "experience_years_clean",
            "location_clean",
            "work_mode_clean",
            "level_clean",
            "salary_target",
        ]
    ].copy()

    return df


def build_reference_features(df: pd.DataFrame):
    X = pd.get_dummies(df.drop(columns=["salary_target"]))
    return X.astype(float)


def map_location_for_model(df: pd.DataFrame, user_location_choice: str) -> str:
    egypt_keywords = [
        "egypt", "cairo", "giza", "alexandria", "6 october", "6th of october",
        "new cairo", "maadi", "heliopolis", "nasr city", "mansoura", "tanta",
        "suez", "ismailia", "zagazig", "sharm", "hurghada", "asyut", "aswan",
        "minya", "sohag", "qena", "fayoum", "banha", "damanhur", "port said",
        "القاهرة", "الجيزة", "الاسكندرية", "الإسكندرية", "اكتوبر", "مصر"
    ]

    location_series = df["location_clean"].dropna().astype(str)

    def is_egypt(loc: str) -> bool:
        txt = loc.strip().lower()
        if txt in ["unknown", "nan", ""]:
            return True
        return any(k in txt for k in egypt_keywords)

    egypt_locations = location_series[location_series.apply(is_egypt)]
    outside_locations = location_series[~location_series.apply(is_egypt)]

    if user_location_choice == "Egypt":
        if not egypt_locations.empty:
            return egypt_locations.value_counts().idxmax()
        return "Cairo"
    else:
        if not outside_locations.empty:
            return outside_locations.value_counts().idxmax()
        return "United Arab Emirates"


def clean_work_mode_options(df: pd.DataFrame):
    vals = sorted(
        x for x in df["work_mode_clean"].dropna().astype(str).unique().tolist()
        if x.strip().lower() != "unknown"
    )
    preferred = ["On-Site", "Hybrid", "Remote"]
    ordered = [x for x in preferred if x in vals] + [x for x in vals if x not in preferred]
    return ordered


def clean_seniority_options(df: pd.DataFrame):
    vals = sorted(
        x for x in df["level_clean"].dropna().astype(str).unique().tolist()
        if x.strip().lower() != "unknown"
    )
    preferred = ["Entry-Level", "Junior", "Mid-Level", "Senior", "Lead/Manager"]
    ordered = [x for x in preferred if x in vals] + [x for x in vals if x not in preferred]
    return ordered


def build_input_row(job_title, experience_years, location_value, work_mode, level, reference_columns):
    input_df = pd.DataFrame(
        [
            {
                "job_title_clean": job_title,
                "experience_years_clean": experience_years,
                "location_clean": location_value,
                "work_mode_clean": work_mode,
                "level_clean": level,
            }
        ]
    )

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=reference_columns, fill_value=0)
    return input_encoded.astype(float)


def get_smoothed_salary(model, reference_columns, job_title, experience_years, location_value, work_mode, level):
    raw_predictions = []

    for exp in range(experience_years + 1):
        input_row = build_input_row(
            job_title=job_title,
            experience_years=exp,
            location_value=location_value,
            work_mode=work_mode,
            level=level,
            reference_columns=reference_columns,
        )

        pred_log = model.predict(input_row)[0]
        pred_salary = np.expm1(pred_log)
        pred_salary = max(0, float(pred_salary))
        raw_predictions.append(pred_salary)

    smoothed = [raw_predictions[0]]

    for i in range(1, len(raw_predictions)):
        minimum_allowed = smoothed[i - 1] + 1000
        smoothed_value = max(raw_predictions[i], minimum_allowed)
        smoothed.append(smoothed_value)

    return smoothed[-1]


if not MODEL_PATH.exists():
    st.error("Optimized model file not found.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Standardized training data file not found.")
    st.stop()

try:
    model = load_model()
    df = load_data()
    X_reference = build_reference_features(df)
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

st.title("Egypt Tech Salary Predictor")

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #eef7ff 0%, #f5f3ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    ">
        <div style="font-size: 25px; font-weight: 800; line-height: 1.6; color: #0f172a;">
            An interactive salary prediction dashboard powered by a machine learning model trained on
            <span style="color:#0f766e;">5,400+ cleaned job salary records</span>
            from the Egyptian tech market and standardized into
            <span style="color:#7c3aed;">11 core tech roles</span>.
        </div>
        <div style="font-size: 17px; color: #475569; margin-top: 10px; line-height: 1.6;">
            Built using cleaned market data, standardized job roles, and an optimized XGBoost model.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

job_titles = sorted(df["job_title_clean"].dropna().astype(str).unique().tolist())
work_modes = clean_work_mode_options(df)
levels = clean_seniority_options(df)

selected_job_title = st.selectbox("Select Position", job_titles)
selected_experience = st.slider("Years of Experience", min_value=0, max_value=15, value=1, step=1)
selected_location_group = st.selectbox("Company Location", ["Egypt", "Outside Egypt"])
selected_work_mode = st.selectbox("Select Work Mode", work_modes)
selected_level = st.selectbox("Select Seniority Level", levels)

if st.button("Predict Salary"):
    try:
        model_location_value = map_location_for_model(df, selected_location_group)

        pred_salary = get_smoothed_salary(
            model=model,
            reference_columns=X_reference.columns,
            job_title=selected_job_title,
            experience_years=selected_experience,
            location_value=model_location_value,
            work_mode=selected_work_mode,
            level=selected_level,
        )

        st.success("Prediction generated successfully.")
        st.metric("Estimated Monthly Salary", f"{pred_salary:,.0f} EGP")

        st.write(
            f"For a **{selected_job_title}** with **{selected_experience} years** of experience, "
            f"at a company based in **{selected_location_group}**, with **{selected_work_mode}** work mode "
            f"and **{selected_level}** seniority level, the estimated monthly salary is "
            f"**{pred_salary:,.0f} EGP**."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("This prediction is based on the optimized model trained on the standardized salary dataset.")