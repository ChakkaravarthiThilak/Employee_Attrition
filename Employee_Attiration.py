import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load models
try:
    with open("C:/Users/Administrator/best_attrition_model.pkl", "rb") as f:
        attrition_model = pickle.load(f)
    with open("C:/Users/Administrator/best_performance_model.pkl", "rb") as f:
        performance_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please check paths.")
    st.stop()

# Custom preprocessing functions
def preprocess_attrition(input_df):
    encoding_maps = {
        'Department': {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
        'MaritalStatus': {'Divorced': 0, 'Married': 1, 'Single': 2},
        'OverTime': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in encoding_maps.items():
        input_df[col] = input_df[col].map(mapping)
    
    numerical_cols = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']
    scaler = StandardScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    return input_df

def preprocess_performance(input_df):
    numerical_cols = ['Education', 'JobInvolvement', 'JobLevel', 
                     'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole']
    
    scaler = StandardScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    return input_df

# Streamlit app structure
st.title("Employee Analytics Predictor")
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select Prediction Type", 
                           ["Attrition Prediction", "Performance Rating Prediction"])

if page == "Attrition Prediction":
    st.header("Employee Attrition Prediction")
    
    # Input section
    with st.form("attrition_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            department = st.selectbox("Department", 
                                    ["Human Resources", "Research & Development", "Sales"])
            income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        with col2:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            marital_status = st.selectbox("Marital Status", 
                                        ["Divorced", "Married", "Single"])
            overtime = st.selectbox("Works Overtime", ["No", "Yes"])
        
        submitted_attr = st.form_submit_button("Predict Attrition")

    if submitted_attr:
        # Create input dataframe
        input_data = pd.DataFrame([[
            age, department, income, job_satisfaction,
            years_at_company, marital_status, overtime
        ]], columns=[
            'Age', 'Department', 'MonthlyIncome', 'JobSatisfaction',
            'YearsAtCompany', 'MaritalStatus', 'OverTime'
        ])
        
        # Preprocess and predict
        processed_data = preprocess_attrition(input_data)
        prediction = attrition_model.predict(processed_data)[0]
        proba = attrition_model.predict_proba(processed_data)[0][1]
        
        # Display results
        st.subheader("Prediction Result")
        result = "Yes" if prediction == 1 else "No"
        color = "red" if prediction == 1 else "green"
        st.markdown(f"**Predicted Attrition**: <span style='color:{color}'>{result}</span>", 
                    unsafe_allow_html=True)
        st.write(f"Probability of Attrition: {proba:.2%}")

elif page == "Performance Rating Prediction":
    st.header("Performance Rating Prediction")
    
    # Input section
    with st.form("performance_form"):
        col1, col2 = st.columns(2)
        with col1:
            education = st.slider("Education Level (1-5)", 1, 5, 3)
            job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
            job_level = st.slider("Job Level (1-5)", 1, 5, 2)
        with col2:
            income_perf = st.number_input("Monthly Income ($)", 
                                        min_value=1000, max_value=20000, value=5000)
            years_company = st.number_input("Years at Company", 
                                          min_value=0, max_value=40, value=5)
            years_role = st.number_input("Years in Current Role", 
                                       min_value=0, max_value=40, value=2)
        
        submitted_perf = st.form_submit_button("Predict Performance Rating")

    if submitted_perf:
        # Create input dataframe
        input_data = pd.DataFrame([[
            education, job_involvement, job_level, income_perf,
            years_company, years_role
        ]], columns=[
            'Education', 'JobInvolvement', 'JobLevel', 'MonthlyIncome',
            'YearsAtCompany', 'YearsInCurrentRole'
        ])
        
        # Preprocess and predict
        processed_data = preprocess_performance(input_data)
        prediction = performance_model.predict(processed_data)[0]
        
        # Display results
        st.subheader("Prediction Result")
        st.markdown(f"**Predicted Performance Rating**: {prediction}")

# Important Notes
st.markdown("---")
st.warning("""
**Important Limitations:**
1. Preprocessing might not match original training due to:
   - Label encoding mappings are assumed
   - Scaling uses new scalers instead of original
2. Model accuracy might be impacted
3. Fix by saving preprocessing objects during training
""")

# Optional styling
st.markdown("""
<style>
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    .stMarkdown {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)
