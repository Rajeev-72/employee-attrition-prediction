import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD MODEL AND DATA
# =========================================================

MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/employee_data.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Check whether files exist
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please check: models/best_model.pkl")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please check: data/employee_data.csv")
    st.stop()

model = load_model()
df = load_data()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Employee Attrition System")

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Home",
        "📈 HR Dashboard",
        "🔮 Attrition Prediction",
        "💡 Retention Recommendations",
        "🤖 Model Performance"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("📊 Employee Attrition Prediction & Retention Analytics System")

    st.write(
        """
        This Machine Learning application predicts whether an employee is likely
        to leave an organization and provides insights to support data-driven
        employee retention decisions.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🤖 Machine Learning")
        st.write("Predict employee attrition using trained ML models.")

    with col2:
        st.info("📈 HR Analytics")
        st.write("Analyze employee attrition patterns and trends.")

    with col3:
        st.info("💡 Retention Insights")
        st.write("Get suggestions for reducing attrition risk.")

    st.divider()

    st.subheader("🔄 Project Workflow")

    st.write(
        """
        Employee Data  
        ↓  
        Data Preprocessing  
        ↓  
        Machine Learning Model  
        ↓  
        Attrition Prediction  
        ↓  
        Risk Classification  
        ↓  
        Retention Recommendations
        """
    )

# =========================================================
# HR DASHBOARD
# =========================================================

elif page == "📈 HR Dashboard":

    st.title("📈 HR Analytics Dashboard")

    # Basic Metrics
    total_employees = len(df)

    if "Attrition" in df.columns:

        attrition_count = (
            df["Attrition"]
            .astype(str)
            .str.lower()
            .eq("yes")
            .sum()
        )

        attrition_rate = (
            attrition_count / total_employees
        ) * 100

    else:
        attrition_count = 0
        attrition_rate = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Employees",
            total_employees
        )

    with col2:
        st.metric(
            "Employees Left",
            attrition_count
        )

    with col3:
        st.metric(
            "Attrition Rate",
            f"{attrition_rate:.2f}%"
        )

    st.divider()

    # Attrition Distribution
    if "Attrition" in df.columns:

        st.subheader("Employee Attrition Distribution")

        attrition_distribution = (
            df["Attrition"]
            .value_counts()
        )

        st.bar_chart(attrition_distribution)

    # Department Analysis
    if "Department" in df.columns and "Attrition" in df.columns:

        st.subheader("Attrition by Department")

        department_data = pd.crosstab(
            df["Department"],
            df["Attrition"]
        )

        st.bar_chart(department_data)

    # Overtime Analysis
    if "OverTime" in df.columns and "Attrition" in df.columns:

        st.subheader("Attrition by Overtime")

        overtime_data = pd.crosstab(
            df["OverTime"],
            df["Attrition"]
        )

        st.bar_chart(overtime_data)

# =========================================================
# ATTRITION PREDICTION
# =========================================================

elif page == "🔮 Attrition Prediction":

    st.title("🔮 Employee Attrition Prediction")

    st.write(
        "Enter employee information to predict the probability of attrition."
    )

    st.divider()

    # Input Layout
    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            value=50000
        )

        years_at_company = st.number_input(
            "Years at Company",
            min_value=0,
            max_value=50,
            value=5
        )

        job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=3
        )

    with col2:

        overtime = st.selectbox(
            "Overtime",
            ["Yes", "No"]
        )

        work_life_balance = st.slider(
            "Work-Life Balance",
            min_value=1,
            max_value=4,
            value=3
        )

        job_level = st.number_input(
            "Job Level",
            min_value=1,
            max_value=5,
            value=2
        )

        total_working_years = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=50,
            value=5
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Attrition",
        use_container_width=True
    )

    if predict_button:

        # Convert input data into DataFrame
        input_data = pd.DataFrame({

            "Age": [age],

            "MonthlyIncome": [monthly_income],

            "YearsAtCompany": [years_at_company],

            "JobSatisfaction": [job_satisfaction],

            "OverTime": [overtime],

            "WorkLifeBalance": [work_life_balance],

            "JobLevel": [job_level],

            "TotalWorkingYears": [total_working_years]

        })

        try:

            # Make Prediction
            prediction = model.predict(input_data)[0]

            # Probability
            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_data
                )[0][1]

            else:

                probability = None

            st.divider()

            # Prediction Result
            if prediction == 1 or str(prediction).lower() == "yes":

                st.error(
                    "⚠️ Prediction: Employee is likely to leave"
                )

                risk_level = "HIGH"

            else:

                st.success(
                    "✅ Prediction: Employee is likely to stay"
                )

                risk_level = "LOW"

            # Probability
            if probability is not None:

                probability_percentage = probability * 100

                st.metric(
                    "Attrition Probability",
                    f"{probability_percentage:.2f}%"
                )

                if probability_percentage < 30:

                    risk_level = "LOW"

                elif probability_percentage < 70:

                    risk_level = "MEDIUM"

                else:

                    risk_level = "HIGH"

            # Risk Level
            st.subheader("Risk Level")

            if risk_level == "HIGH":

                st.error("🔴 HIGH ATTRITION RISK")

            elif risk_level == "MEDIUM":

                st.warning("🟠 MEDIUM ATTRITION RISK")

            else:

                st.success("🟢 LOW ATTRITION RISK")

        except Exception as e:

            st.error(
                "Prediction failed. The input columns may not match the columns used during model training."
            )

            st.exception(e)

# =========================================================
# RETENTION RECOMMENDATIONS
# =========================================================

elif page == "💡 Retention Recommendations":

    st.title("💡 Employee Retention Recommendations")

    st.write(
        """
        These recommendations are based on common factors associated with
        employee attrition.
        """
    )

    st.divider()

    st.subheader("Possible Retention Strategies")

    st.write(
        """
        🔹 **High Overtime**

        Recommendation:
        Review workload and improve work-life balance.

        ---

        🔹 **Low Job Satisfaction**

        Recommendation:
        Conduct employee feedback sessions and identify workplace concerns.

        ---

        🔹 **Low Income**

        Recommendation:
        Consider compensation review and career growth opportunities.

        ---

        🔹 **Limited Career Growth**

        Recommendation:
        Provide training, promotion opportunities, and career development plans.

        ---

        🔹 **Poor Work-Life Balance**

        Recommendation:
        Review working hours and workload distribution.
        """
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write(
        """
        The final model should be evaluated using multiple classification
        metrics rather than accuracy alone.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "Add Result")

    with col2:
        st.metric("Precision", "Add Result")

    with col3:
        st.metric("Recall", "Add Result")

    with col4:
        st.metric("F1 Score", "Add Result")

    st.divider()

    st.subheader("Algorithms Compared")

    models = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ],

        "Status": [
            "Compared",
            "Compared",
            "Compared",
            "Compared"
        ]

    })

    st.table(models)

    st.info(
        "The final model should be selected based on evaluation metrics and the project objective."
    )
