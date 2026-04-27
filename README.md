# Egypt Tech Salary Predictor

A Big Data / Machine Learning project that predicts expected monthly salaries for tech jobs in Egypt using real-world job market data collected from multiple sources and prepared through a complete data pipeline.

---

## Project Overview

This project started with scraping job postings from Wuzzuf to explore the Egyptian tech job market, then evolved into a full salary prediction system based on:

- cleaned and merged salary datasets
- standardized tech job categories
- years of experience
- location grouping
- work mode
- seniority level

The final system provides an interactive Streamlit dashboard where the user selects a tech position and job-related attributes, then receives an estimated monthly salary in EGP.

---

## Main Objectives

- Collect real tech job data from the Egyptian market
- Clean and normalize salary and job information
- Standardize job titles into a limited set of tech categories
- Build intermediate datasets for training and dashboard usage
- Train machine learning models for salary prediction
- Deploy the final model using Streamlit

---

## Final Job Categories

The final project standardizes jobs into the following categories:

- back end engineer
- front end engineer
- full stack engineer
- data/ai engineer
- software testing engineer
- mobile engineer
- devops engineer
- embedded engineer
- technical support engineer
- cybersecurity engineer
- ui/ux designer

---

## Final Dashboard Inputs

The final optimized dashboard uses:

- `job_title_clean`
- `experience_years_clean`
- `location_clean` (simplified in the dashboard for usability)
- `work_mode_clean`
- `level_clean`

Target:

- `salary_target`

---

## Data Pipeline

The project includes multiple stages of data preparation.

### 1. Wuzzuf Scraping Pipeline

Inside `wuzzuf_scraping_pipeline/`:

- `scrape.py`  
  Scrapes job postings from Wuzzuf using Selenium.

- `data_cleaning.py`  
  Cleans salary and experience fields and prepares structured data.

- `model_training.py`  
  Trains the original Wuzzuf-only salary model.

This part is preserved in the repository because it represents the original scraping and preprocessing work required for the project.

### 2. Data Preparation and Merging

Inside `scripts/`:

- `create_master_schema.py`  
  Builds the schema template used for combining sources.

- `merge_all_training_files.py`  
  Merges multiple prepared datasets into one training base.

- `normalize_kaggle_generic.py`  
  Normalizes external generic salary datasets.

- `normalize_wuzzuf_training.py`  
  Normalizes the Wuzzuf training data.

- `prepare_final_model_data.py`  
  Builds the training-ready dataset for the baseline model.

- `standardize_job_titles.py`  
  Standardizes raw job titles into the final 11 tech categories.

- `build_final_dashboard_data.py`  
  Builds the final cleaned dataset used for dashboard-oriented modeling.

### 3. Model Training

Inside `scripts/`:

- `train_final_model.py`  
  Trains the simpler baseline model.

- `train_optimized_model.py`  
  Trains the optimized XGBoost model used by the final dashboard.

---

## Important Datasets

Inside `data_sources/merged/`:

- `master_salary_data.csv`  
  The merged raw salary dataset.

- `ml_ready_salary_data.csv`  
  The cleaned dataset prepared for machine learning.

- `ml_ready_salary_data_standardized.csv`  
  The dataset after standardizing job titles.

- `final_dashboard_ready_data.csv`  
  The final cleaned dataset prepared for dashboard-oriented prediction.

- `final_model_training_data.csv`  
  The dataset used in the earlier baseline pipeline.

- `master_schema_template.csv`  
  The schema template used during preparation.

---

## Models

Inside `salary_prediction_model/`:

- `final_salary_model.pkl`  
  The earlier baseline model.

- `optimized_salary_model.pkl`  
  The final optimized model used in the deployed dashboard.

- `final_model_metrics.json`  
  Metrics for the earlier baseline model.

---

## Final Optimized Model

The final deployed version uses:

- **XGBoost Regressor**

### Final Optimized Evaluation

- **R² = 0.5286**
- **MAE = 11,992 EGP**
- **RMSE = 23,158 EGP**

These results were achieved after:
- standardizing job titles
- cleaning salary outliers
- keeping only relevant tech categories
- using multiple structured input features instead of only job title and experience

---

## Dashboard

The final Streamlit dashboard is:

- `optimized_dashboard.py`

It allows the user to:

- choose one of the standardized tech job categories
- choose years of experience
- choose company location
- choose work mode
- choose seniority level
- receive an estimated monthly salary in EGP

For usability, the dashboard simplifies some raw dataset values while still using the trained optimized model underneath.

---

## Project Structure

```bash
egypt-tech-salary-predictor/
│
├── data_sources/
│   ├── raw/
│   ├── processed/
│   ├── training/
│   └── merged/
│       ├── final_dashboard_ready_data.csv
│       ├── final_model_training_data.csv
│       ├── master_salary_data.csv
│       ├── master_schema_template.csv
│       ├── ml_ready_salary_data.csv
│       └── ml_ready_salary_data_standardized.csv
│
├── reports/
│   ├── optimized_drivers.png
│   └── optimized_performance.png
│
├── salary_prediction_model/
│   ├── final_salary_model.pkl
│   └── optimized_salary_model.pkl
│
├── scripts/
│   ├── build_final_dashboard_data.py
│   ├── create_master_schema.py
│   ├── merge_all_training_files.py
│   ├── normalize_kaggle_generic.py
│   ├── normalize_wuzzuf_training.py
│   ├── prepare_final_model_data.py
│   ├── standardize_job_titles.py
│   ├── train_final_model.py
│   └── train_optimized_model.py
│
├── wuzzuf_scraping_pipeline/
│   ├── scrape.py
│   ├── data_cleaning.py
│   └── model_training.py
│
├── optimized_dashboard.py
├── README.md
├── requirements.txt
└── .gitignore
How to Run the Project
1. Clone the repository
git clone https://github.com/youssefehab241/egypt-tech-salary-predictor.git
cd egypt-tech-salary-predictor
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment
Windows PowerShell
.\venv\Scripts\Activate.ps1
Windows CMD
venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Run the final dashboard
python -m streamlit run optimized_dashboard.py
Notes
The final deployed dashboard uses:
salary_prediction_model/optimized_salary_model.pkl
The project also keeps the older baseline model for comparison.
The Wuzzuf scraping pipeline is preserved for academic review and grading.
This project does not use a scraping API for Wuzzuf; it uses Selenium-based browser automation.
The repository keeps intermediate datasets intentionally to show the full data engineering pipeline.
Future Improvements

Possible next steps:

improve location handling with cleaner geographic grouping
refine seniority labeling further
test more feature engineering approaches
compare XGBoost with CatBoost or LightGBM
deploy the final dashboard publicly
improve dashboard design and user experience
Demo

A demo video is included separately to explain:

the project idea
the data pipeline
the model training process
the final dashboard