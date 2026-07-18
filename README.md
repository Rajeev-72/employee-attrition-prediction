# 📊 Employee Attrition Prediction System

An end-to-end machine learning system that predicts which employees are at risk of
leaving, explains *why* using SHAP, classifies them into risk tiers, and recommends
concrete retention actions — packaged with a Streamlit HR dashboard and a REST API
for production integration.

> **Built as a complete MLOps-style portfolio project**, following the full
> lifecycle: problem definition → data → modeling → explainability → deployment.

---

## 🧭 Project Workflow

```
Problem Definition → Dataset Collection → Data Understanding → Data Cleaning
→ EDA → Feature Engineering → Preprocessing → Model Training → Model Comparison
→ Best Model Selection → Explainable AI → Prediction → Risk Classification
→ Retention Recommendation → HR Dashboard → Testing → Deployment
```

## 1. Problem Definition

Employee attrition costs companies 6–9 months of a departing employee's salary in
rehiring and lost productivity. HR teams typically find out someone is leaving only
when they hand in their notice — too late to act. This project reframes attrition as
a **supervised binary classification problem**: given an employee's demographic,
compensation, and satisfaction data, predict the probability they will leave, and
turn that probability into an actionable HR intervention *before* they resign.

**Goals:** (1) predict attrition risk with strong recall on the minority "leaver"
class, (2) explain predictions in HR-interpretable terms, (3) ship it as a usable
tool (dashboard + API), not just a notebook.

## 2. Dataset Collection

`src/data_generation.py` generates a **synthetic dataset of 1,500 employees** using
the same 35-column schema as the well-known IBM Watson Analytics "HR Employee
Attrition" sample (Age, Department, OverTime, JobSatisfaction, MonthlyIncome, etc.).
Attrition labels are simulated via a logistic function of real-world risk drivers
(overtime, low satisfaction, low pay relative to level, long commute, frequent
travel, job-hopping history), producing a realistic **16.6% attrition rate**.

> Because the original Kaggle/IBM file is a third-party sample dataset, it isn't
> redistributed here — but this pipeline is schema-compatible. **To use the real
> data instead:** download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from Kaggle and
> drop it in at `data/raw/employee_attrition.csv`, then run the pipeline from
> Step 3 onward.

## 3. Data Understanding & 4. Data Cleaning

`src/data_cleaning.py` profiles the raw data (shape, dtypes, missingness,
duplicates, target balance — see `reports/data_understanding.md`) and then:
- Drops exact duplicate rows
- Median/mode-imputes any missing values (never drops rows — attrition data is
  precious and imbalanced, so row-dropping would bias the minority class further)
- Drops constant/non-informative columns (`EmployeeCount`, `Over18`, `StandardHours`)
- Clips out-of-range values defensively (guards against a real-world CSV swap)
- Adds a binary `Attrition_Flag` target column

## 5. Exploratory Data Analysis

`src/eda.py` produces 8 visualizations (`reports/figures/`) and a written insights
report (`reports/eda_summary.md`). Key findings from this dataset:

| Insight | Finding |
|---|---|
| Overtime | Employees working overtime leave at **26.2%** vs **13.1%** for those who don't |
| Income | Median income of leavers ($7,252) is notably lower than stayers ($10,808) |
| Age | Leavers skew younger (median 25) vs stayers (median 42) |
| Top correlations | `Age`, `TotalWorkingYears`, `YearsAtCompany`, `MonthlyIncome`, `YearsInCurrentRole` |

<p align="center">
  <img src="reports/figures/02_attrition_by_overtime.png" width="45%">
  <img src="reports/figures/06_correlation_heatmap.png" width="45%">
</p>

## 6. Feature Engineering

`src/feature_engineering.py` adds 11 engineered features that outperform raw columns
individually, including: `SatisfactionScore` (composite of 4 satisfaction
dimensions), `TenureRatio`, `IncomeToLevelRatio` (are they underpaid for their
level?), `PromotionStagnation`, `JobHoppingIndex`, and risk flags like
`IsFrequentTraveler` / `HasNoStockOptions` / `LongCommute`.

## 7. Data Preprocessing

`src/preprocessing.py`: label-encodes categoricals, standard-scales all features,
does a stratified 80/20 train/test split, then applies **SMOTE** to the training set
only (test set is left untouched/realistic) — because raw attrition data is
imbalanced (~5:1 stay:leave), which would otherwise bias models toward always
predicting "stays."

## 8. Model Training & 9. Model Comparison

Six classifiers were trained and evaluated on the same held-out test set:

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.810 | 0.455 | **0.700** | 0.551 | **0.855** |
| XGBoost | 0.847 | 0.548 | 0.460 | 0.500 | 0.839 |
| SVM (RBF) | 0.847 | 0.540 | 0.540 | 0.540 | 0.834 |
| Gradient Boosting | 0.833 | 0.500 | 0.460 | 0.479 | 0.830 |
| Random Forest | 0.823 | 0.474 | 0.540 | 0.505 | 0.824 |
| KNN | 0.633 | 0.300 | 0.900 | 0.450 | 0.794 |

*(full run: `reports/model_comparison.csv`, ROC curves: `reports/figures/09_roc_curves.png`)*

## 10. Best Model Selection

**Logistic Regression was selected** (ranked by ROC-AUC — the standard metric for
imbalanced classification since it's threshold-independent) with **0.855 ROC-AUC**
and **0.70 recall** on the leaver class — i.e., it correctly flags 70% of employees
who actually leave, which matters more than raw accuracy for a retention use case
where missing a flight-risk employee is costlier than a false alarm.
`src/model_comparison.py` auto-promotes the winner to `models/best_model.joblib`.

> **Why does Logistic Regression win here?** Because this project's synthetic
> labels were generated from a logistic function of the risk factors (by design, to
> keep the ground truth interpretable and auditable). On real-world HR data,
> non-linear models like XGBoost/Random Forest typically edge out linear models once
> feature interactions get messier — the comparison harness handles either case
> automatically, whichever dataset you plug in.

## 11. Explainable AI

`src/explainability.py` uses **SHAP** to explain the winning model both globally and
per-employee. Top global drivers: `Age`, `TotalWorkingYears`, `YearsInCurrentRole`,
`OverTime`, `JobLevel`, `IncomeToLevelRatio`, `JobSatisfaction`.

<p align="center">
  <img src="reports/figures/12_shap_summary.png" width="60%">
</p>

## 12. Attrition Prediction, 13. Risk Classification & 14. Retention Recommendation

`src/predict.py` (`AttritionPredictor` class) runs new employee records through the
full saved pipeline end-to-end and returns:
- **`AttritionProbability`** — model's predicted probability of leaving
- **`RiskTier`** — Low (`<30%`) / Medium (`30–60%`) / High (`≥60%`)
- **`RecommendedAction`** — a rule-based engine maps the employee's *own* triggered
  risk factors (overtime, low satisfaction, no equity, stagnant promotion, long
  commute, job-hopping pattern, etc.) to the top 3 most relevant HR interventions

```bash
python src/predict.py --input data/raw/employee_attrition.csv --output reports/predictions.csv
```

## 15. HR Analytics Dashboard

An interactive **Streamlit** dashboard (`dashboard/app.py`) for non-technical HR
users: KPI overview, filterable employee risk table, workforce trend charts,
SHAP explainability tab, and a prioritized retention-action worklist.

```bash
streamlit run dashboard/app.py
```

## 16. Testing

22 automated tests (`pytest`) cover data generation sanity checks, cleaning
invariants, feature engineering correctness, risk-tier threshold logic, the
recommendation engine, and end-to-end prediction output shape/range.

```bash
pytest tests/ -v      # 22 passed
```

## 17. Deployment

Two deployment paths are included:
- **REST API** (`api/app.py`, Flask + Gunicorn) — `POST /predict` with a JSON list
  of employee records, returns risk scores + recommendations. Ideal for
  integrating into an existing HRIS.
- **Docker** (`Dockerfile`) — builds the full pipeline at image build time and
  serves the API with Gunicorn.

```bash
docker build -t attrition-api .
docker run -p 5000:5000 attrition-api
```

The dashboard can also be deployed for free on **Streamlit Community Cloud** by
pointing it at `dashboard/app.py` in this repo.

## 18. Documentation

See [`docs/documentation.md`](docs/documentation.md) for a deeper technical write-up
of each pipeline stage, design decisions, and trade-offs.

## 19. Resume & Interview Preparation

See [`docs/RESUME_AND_INTERVIEW_PREP.md`](docs/RESUME_AND_INTERVIEW_PREP.md) for
ready-to-use resume bullet points and likely interview questions/answers for this
project.

---

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/employee-attrition-prediction.git
cd employee-attrition-prediction
pip install -r requirements.txt

python run_pipeline.py          # runs steps 2–14 end-to-end (~20s)
pytest tests/ -v                # steps 16: 22 tests
streamlit run dashboard/app.py  # step 15: interactive dashboard
python api/app.py               # step 17: REST API on :5000
```

## 📁 Project Structure

```
employee-attrition-prediction/
├── data/raw/                  # dataset
├── data/processed/            # cleaned/engineered/split data (regenerated)
├── src/                       # pipeline: generation, cleaning, EDA, features,
│                               preprocessing, training, comparison, SHAP, predict
├── models/                    # trained models + best_model.joblib + preprocessor
├── reports/                   # EDA figures, model comparison table, predictions
├── dashboard/app.py           # Streamlit HR dashboard
├── api/app.py                 # Flask REST API
├── tests/                     # pytest suite
├── docs/                      # technical docs + resume/interview prep
├── run_pipeline.py            # one-command full pipeline runner
├── Dockerfile / requirements.txt
```

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP ·
Streamlit · Plotly · Flask · Gunicorn · Docker · pytest

## 📄 License

MIT — see [LICENSE](LICENSE).
