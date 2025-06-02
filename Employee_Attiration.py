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
st.set_page_config(
    page_title="Employee Analytics Dashboard",
    page_icon="👥",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
                       ["Home", "Attrition Prediction", "Performance Rating Prediction"])
st.sidebar.markdown("---")
st.sidebar.info(
    "This application helps predict employee attrition and performance ratings "
    "based on HR analytics data."
)

# Home Page
if page == "Home":
    st.title("Employee Analytics Dashboard")
    st.subheader("Predict Employee Attrition and Performance Ratings")
    
    st.image("https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80",
             use_column_width=True, caption="Employee Analytics")
    
    st.markdown("""
    ## Welcome to the Employee Analytics Prediction System
    
    This application uses machine learning models to predict two key HR metrics:
    
    ### 1. Employee Attrition Prediction
    Predict whether an employee is likely to leave the company based on:
    - Demographic factors (Age, Marital Status)
    - Employment details (Department, Years at Company)
    - Job satisfaction metrics
    - Compensation (Monthly Income)
    
    ### 2. Performance Rating Prediction
    Predict an employee's performance rating based on:
    - Education level
    - Job involvement
    - Position level
    - Compensation
    - Tenure metrics
    
    ## How to Use
    1. Select the desired prediction page from the sidebar
    2. Fill in the employee details using the input forms
    3. Click the prediction button to get results
    
    ## Benefits
    - Identify at-risk employees for retention efforts
    - Recognize high-potential employees for development programs
    - Make data-driven HR decisions
    - Improve workforce planning and management
    
    """)
    
    # Display model information
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Model")
        st.write("Algorithm: Random Forest Classifier")
        st.write("Accuracy: ~85%")
        st.write("Key Features: Age, Income, Job Satisfaction, Overtime")
        
    with col2:
        st.subheader("Performance Model")
        st.write("Algorithm: Random Forest Classifier")
        st.write("Accuracy: ~82%")
        st.write("Key Features: Education, Job Level, Income, Tenure")
    
    st.warning("""
    **Important Note:** 
    Predictions are based on historical patterns and should be used as one input 
    among many in HR decision-making. Always consider individual circumstances 
    and context when interpreting results.
    """)

# Attrition Prediction Page
elif page == "Attrition Prediction":
    st.title("Employee Attrition Prediction")
    st.info("Predict whether an employee is likely to leave the company")
    
    # Input section
    with st.form("attrition_form"):
        st.subheader("Employee Details")
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
        
        if prediction == 1:
            st.error(f"⚠️ High Attrition Risk: {proba:.2%} probability")
            st.markdown("""
            **Recommended Actions:**
            - Schedule retention interview
            - Review compensation package
            - Consider career development opportunities
            - Evaluate workload balance
            """)
        else:
            st.success(f"✅ Low Attrition Risk: {proba:.2%} probability")
            st.markdown("""
            **Recommended Actions:**
            - Continue regular engagement
            - Monitor job satisfaction
            - Provide growth opportunities
            """)
        
        # Show probability gauge
        st.progress(proba)
        st.caption(f"Probability of attrition: {proba:.2%}")

# Performance Rating Prediction Page
elif page == "Performance Rating Prediction":
    st.title("Performance Rating Prediction")
    st.info("Predict an employee's performance rating (1-4 scale)")
    
    # Input section
    with st.form("performance_form"):
        st.subheader("Employee Details")
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
        
        # Performance rating visualization
        rating_emoji = {
            1: "⭐",
            2: "⭐⭐",
            3: "⭐⭐⭐",
            4: "⭐⭐⭐⭐"
        }.get(prediction, "❓")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Predicted Rating", f"{prediction} {rating_emoji}")
        with col2:
            rating_text = {
                1: "Needs Improvement",
                2: "Meets Expectations",
                3: "Exceeds Expectations",
                4: "Outstanding"
            }.get(prediction, "Unknown")
            st.subheader(rating_text)
        
        # Interpretation and recommendations
        st.markdown("""
        ### Performance Insights
        """)
        
        if prediction >= 3:
            st.success("**High Performer Detected**")
            st.markdown("""
            - Consider for leadership development programs
            - Review for promotion opportunities
            - Assign challenging projects
            - Recognize achievements
            """)
        else:
            st.warning("**Development Opportunity**")
            st.markdown("""
            - Create development plan
            - Provide coaching/mentoring
            - Set clear performance goals
            - Schedule regular check-ins
            """)

# Footer
st.markdown("---")
st.caption("Employee Analytics Dashboard v1.0 | HR Analytics Team")

# Add some styling
st.markdown("""
<style>
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    .stMarkdown {
        margin-top: 1rem;
    }
    .st-b7 {
        background-color: #f0f2f6;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stProgress > div > div > div {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)
