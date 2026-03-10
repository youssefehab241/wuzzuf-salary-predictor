import streamlit as st
import pandas as pd
import joblib

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction Platform",
    page_icon="💰",
    layout="centered"
)

# --- 2. Load the Trained Model ---
# @st.cache_resource ensures the model is loaded only once to save memory and time
@st.cache_resource
def load_model():
    return joblib.load('salary_prediction_model.pkl')

model = load_model()

# --- 3. Dashboard Header ---
st.title("📊 Job Market Salary Prediction Platform")
st.write("Predict the estimated salary for software and tech jobs in the Egyptian market based on real Wuzzuf data.")
st.markdown("---")

# --- 4. User Input Form ---
with st.form("prediction_form"):
    st.subheader("🔍 Enter Job Details")
    
    # Input fields
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Python Developer")
    skills = st.text_input("Required Skills", placeholder="e.g., Python, Django, SQL, REST APIs")
    experience = st.number_input("Minimum Experience (Years)", min_value=0, max_value=30, value=2)
    
    # Submit button
    submit_button = st.form_submit_button(label="🔮 Predict Salary")

# --- 5. Prediction Logic ---
if submit_button:
    if job_title.strip() == "":
        st.warning("⚠️ Please enter a Job Title.")
    else:
        # Format the user input exactly how the model expects it
        input_data = pd.DataFrame({
            'Title': [job_title],
            'Skills': [skills],
            'Min_Experience_Years': [experience]
        })
        
        # Make the prediction
        with st.spinner("🤖 AI is analyzing market data..."):
            prediction = model.predict(input_data)[0]
            
        # Display the result beautifully
        st.success("Prediction Complete!")
        st.metric(label="Estimated Average Salary", value=f"{prediction:,.0f} EGP")
        
        st.info("💡 Note: This prediction is based on scraped market data. Real salaries may vary based on company size and negotiation skills.")