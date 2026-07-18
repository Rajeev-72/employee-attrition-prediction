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

# FILE PATHS

# =========================================================

MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
DATA_PATH = "data/processed/cleaned_data.csv"

# =========================================================

# LOAD MODEL

# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# =========================================================

# CHECK FILES

# =========================================================

if not os.path.exists(MODEL_PATH):
st.error(
"Model file not found. Please check: models/best_model.joblib"
)
st.stop()

if not os.path.exists(PREPROCESSOR_PATH):
st.error(
"Preprocessor file not found. Please check: models/preprocessor.joblib"
)
st.stop()

if not os.path.exists(DATA_PATH):
st.error(
"Dataset not found. Please check: data/processed/cleaned_data.csv"
)
st.stop()

# =========================================================

# LOAD FILES

# =========================================================

model = load_model()
preprocessor = load_preprocessor()
df = load_data()

# =========================================================

# FEATURE ENGINEERING

# =========================================================

def engineer_features(input_df, reference_df=None):

```
input_df = input_df.copy()

# 1. Tenure Ratio
input_df["TenureRatio"] = (
    input_df["YearsAtCompany"]
    /
    input_df["TotalWorkingYears"].replace(0, 1)
)

# 2. Job Hopping Index
input_df["JobHoppingIndex"] = (
    input_df["NumCompaniesWorked"]
    /
    input_df["TotalWorkingYears"].replace(0, 1)
)

# 3. Satisfaction Score
input_df["SatisfactionScore"] = input_df[
    [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance"
    ]
].mean(axis=1)

# 4. Income to Level Ratio
# Use the original dataset to calculate average income
# for each JobLevel, exactly like during training.

if reference_df is not None:

    level_avg_income = (
        reference_df
        .groupby("JobLevel")["MonthlyIncome"]
        .mean()
    )

    input_df["IncomeToLevelRatio"] = (
        input_df["MonthlyIncome"]
        /
        input_df["JobLevel"].map(level_avg_income)
    )

else:

    input_df["IncomeToLevelRatio"] = 1.0

# 5. Promotion Stagnation
input_df["PromotionStagnation"] = (
    input_df["YearsSinceLastPromotion"]
    /
    input_df["YearsAtCompany"].replace(0, 1)
)

# 6. Manager Stability
input_df["ManagerStability"] = (
    input_df["YearsWithCurrManager"]
    /
    input_df["YearsAtCompany"].replace(0, 1)
)

# 7. Frequent Traveler
input_df["IsFrequentTraveler"] = (
    input_df["BusinessTravel"]
    ==
    "Travel_Frequently"
).astype(int)

# 8. No Stock Options
input_df["HasNoStockOptions"] = (
    input_df["StockOptionLevel"] == 0
).astype(int)

# 9. Early Career
input_df["IsEarlyCareer"] = (
    input_df["TotalWorkingYears"] <= 3
).astype(int)

# 10. Long Commute
input_df["LongCommute"] = (
    input_df["DistanceFromHome"] >= 15
).astype(int)

# 11. Age Group
input_df["AgeGroup"] = pd.cut(
    input_df["Age"],
    bins=[17, 25, 35, 45, 55, 65],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-60"
    ]
)

# Clean infinite values
input_df = input_df.replace(
    [float("inf"), float("-inf")],
    0
)

# Fill missing ratio values
ratio_cols = [
    "TenureRatio",
    "JobHoppingIndex",
    "IncomeToLevelRatio",
    "PromotionStagnation",
    "ManagerStability"
]

for column in ratio_cols:

    input_df[column] = (
        input_df[column]
        .fillna(0)
    )

return input_df
```

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

```
st.title(
    "📊 Employee Attrition Prediction & Retention Analytics System"
)

st.write(
    """
    This Machine Learning application predicts whether an employee
    is likely to leave an organization and provides insights to support
    data-driven employee retention decisions.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.info("🤖 Machine Learning")

    st.write(
        "Predict employee attrition using trained ML models."
    )

with col2:

    st.info("📈 HR Analytics")

    st.write(
        "Analyze employee attrition patterns and trends."
    )

with col3:

    st.info("💡 Retention Insights")

    st.write(
        "Get suggestions for reducing attrition risk."
    )

st.divider()

st.subheader("🔄 Project Workflow")

st.write(
    """
    Employee Data
    ↓
    Data Preprocessing
    ↓
    Feature Engineering
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
```

# =========================================================

# HR DASHBOARD

# =========================================================

elif page == "📈 HR Dashboard":

```
st.title("📈 HR Analytics Dashboard")

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
        attrition_count
        /
        total_employees
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

    st.subheader(
        "Employee Attrition Distribution"
    )

    attrition_distribution = (
        df["Attrition"]
        .value_counts()
    )

    st.bar_chart(
        attrition_distribution
    )

# Department Analysis

if (
    "Department" in df.columns
    and
    "Attrition" in df.columns
):

    st.subheader(
        "Attrition by Department"
    )

    department_data = pd.crosstab(
        df["Department"],
        df["Attrition"]
    )

    st.bar_chart(
        department_data
    )

# Overtime Analysis

if (
    "OverTime" in df.columns
    and
    "Attrition" in df.columns
):

    st.subheader(
        "Attrition by Overtime"
    )

    overtime_data = pd.crosstab(
        df["OverTime"],
        df["Attrition"]
    )

    st.bar_chart(
        overtime_data
    )
```

# =========================================================

# ATTRITION PREDICTION

# =========================================================

elif page == "🔮 Attrition Prediction":

```
st.title(
    "🔮 Employee Attrition Prediction"
)

st.write(
    "Enter complete employee information to predict attrition risk."
)

st.divider()

# =====================================================
# INPUTS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("👤 Personal Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    education = st.slider(
        "Education",
        min_value=1,
        max_value=5,
        value=3
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=100,
        value=10
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:

    st.subheader("⭐ Job Satisfaction")

    job_involvement = st.slider(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    environment_satisfaction = st.slider(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    relationship_satisfaction = st.slider(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    work_life_balance = st.slider(
        "Work-Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )

    performance_rating = st.slider(
        "Performance Rating",
        min_value=1,
        max_value=4,
        value=3
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=20,
        value=2
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=50,
        value=5
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=50,
        value=5
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=2
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=20,
        value=1
    )

with col3:

    st.subheader("💼 Employment Information")

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=2
    )

    training_times_last_year = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=20,
        value=3
    )

    stock_option_level = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=1
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "Yes",
            "No"
        ]
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=0,
        max_value=100,
        value=15
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=200000,
        value=5000
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=0,
        max_value=2000,
        value=800
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=0,
        max_value=200,
        value=65
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=0,
        max_value=30000,
        value=15000
    )

st.divider()

predict_button = st.button(
    "🔮 Predict Attrition",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    try:

        # ---------------------------------------------
        # STEP 1: CREATE RAW INPUT DATA
        # ---------------------------------------------

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender],

            "MaritalStatus": [marital_status],

            "Department": [department],

            "EducationField": [education_field],

            "Education": [education],

            "BusinessTravel": [business_travel],

            "DistanceFromHome": [distance_from_home],

            "JobRole": [job_role],

            "JobLevel": [job_level],

            "JobInvolvement": [job_involvement],

            "JobSatisfaction": [job_satisfaction],

            "EnvironmentSatisfaction": [
                environment_satisfaction
            ],

            "RelationshipSatisfaction": [
                relationship_satisfaction
            ],

            "WorkLifeBalance": [
                work_life_balance
            ],

            "PerformanceRating": [
                performance_rating
            ],

            "NumCompaniesWorked": [
                num_companies_worked
            ],

            "TotalWorkingYears": [
                total_working_years
            ],

            "YearsAtCompany": [
                years_at_company
            ],

            "YearsInCurrentRole": [
                years_in_current_role
            ],

            "YearsSinceLastPromotion": [
                years_since_last_promotion
            ],

            "YearsWithCurrManager": [
                years_with_curr_manager
            ],

            "TrainingTimesLastYear": [
                training_times_last_year
            ],

            "StockOptionLevel": [
                stock_option_level
            ],

            "OverTime": [
                overtime
            ],

            "PercentSalaryHike": [
                percent_salary_hike
            ],

            "MonthlyIncome": [
                monthly_income
            ],

            "DailyRate": [
                daily_rate
            ],

            "HourlyRate": [
                hourly_rate
            ],

            "MonthlyRate": [
                monthly_rate
            ]

        })

        # ---------------------------------------------
        # STEP 2: FEATURE ENGINEERING
        # ---------------------------------------------

        input_data = engineer_features(
            input_data,
            reference_df=df
        )

        # ---------------------------------------------
        # STEP 3: APPLY SAVED ENCODERS
        # ---------------------------------------------

        for column in preprocessor["cat_cols"]:

            encoder = (
                preprocessor["encoders"][column]
            )

            input_data[column] = (
                input_data[column]
                .astype(str)
            )

            try:

                input_data[column] = (
                    encoder.transform(
                        input_data[column]
                    )
                )

            except ValueError:

                st.error(
                    f"Unknown category found in column: {column}"
                )

                st.stop()

        # ---------------------------------------------
        # STEP 4: SELECT EXACT FEATURE ORDER
        # ---------------------------------------------

        input_data = input_data[
            preprocessor["feature_cols"]
        ]

        # ---------------------------------------------
        # STEP 5: APPLY STANDARD SCALER
        # ---------------------------------------------

        input_scaled = (
            preprocessor["scaler"]
            .transform(input_data)
        )

        # ---------------------------------------------
        # STEP 6: MAKE PREDICTION
        # ---------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]

        # ---------------------------------------------
        # STEP 7: PREDICTION PROBABILITY
        # ---------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = (
                model
                .predict_proba(input_scaled)[0][1]
            )

        else:

            probability = None

        st.divider()

        # ---------------------------------------------
        # STEP 8: DISPLAY PREDICTION
        # ---------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Employee is likely to leave"
            )

        else:

            st.success(
                "✅ Prediction: Employee is likely to stay"
            )

        # ---------------------------------------------
        # STEP 9: RISK LEVEL
        # ---------------------------------------------

        if probability is not None:

            probability_percentage = (
                probability * 100
            )

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

            st.subheader("Risk Level")

            if risk_level == "HIGH":

                st.error(
                    "🔴 HIGH ATTRITION RISK"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    "🟠 MEDIUM ATTRITION RISK"
                )

            else:

                st.success(
                    "🟢 LOW ATTRITION RISK"
                )

    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(error)
```

# =========================================================

# RETENTION RECOMMENDATIONS

# =========================================================

elif page == "💡 Retention Recommendations":

```
st.title(
    "💡 Employee Retention Recommendations"
)

st.write(
    """
    These recommendations are based on common factors associated
    with employee attrition.
    """
)

st.divider()

st.subheader(
    "Possible Retention Strategies"
)

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
```

# =========================================================

# MODEL PERFORMANCE

# =========================================================

elif page == "🤖 Model Performance":

```
st.title(
    "🤖 Machine Learning Model Performance"
)

st.write(
    """
    The final model should be evaluated using multiple classification
    metrics rather than accuracy alone.
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Accuracy",
        "Add Result"
    )

with col2:

    st.metric(
        "Precision",
        "Add Result"
    )

with col3:

    st.metric(
        "Recall",
        "Add Result"
    )

with col4:

    st.metric(
        "F1 Score",
        "Add Result"
    )

st.divider()

st.subheader(
    "Algorithms Compared"
)

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
```
