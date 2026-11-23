# Customer Churn & Retention Analysis

An end-to-end Data Science and Analytics project focused on understanding customer churn, analyzing retention patterns, identifying high-risk customer segments, and evaluating machine learning models to predict customer churn.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project analyzes customer behavior and identifies factors associated with customer attrition.

The project combines data cleaning, exploratory data analysis (EDA), customer segmentation, retention analysis, machine learning, and an automated interactive HTML dashboard.

## Objectives

- Analyze customer churn patterns and retention trends.
- Identify customer segments with higher churn rates.
- Understand the impact of contract types, payment methods, and tenure on churn.
- Estimate monthly recurring revenue at risk due to churn.
- Build and evaluate machine learning models for churn prediction.
- Present business insights through an interactive HTML dashboard.

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model Storage | Joblib |
| Dashboard | HTML, Plotly |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains customer information, including:

- Customer demographics
- Tenure and contract details
- Internet and additional services
- Monthly and total charges
- Customer churn status

**Total customer records analyzed:** 7,043

## Project Workflow

### 1. Data Understanding & Cleaning
- Explored dataset structure and statistical information.
- Handled missing values in `TotalCharges`.
- Converted columns into appropriate data types.
- Checked and removed duplicate records.
- Exported the cleaned dataset.

### 2. Exploratory Data Analysis
- Analyzed overall churn distribution.
- Examined churn across contract types.
- Studied the relationship between tenure and churn.
- Analyzed churn by Internet Service and Payment Method.
- Compared monthly charges across churn groups.

### 3. Customer Retention Analysis
- Grouped customers based on tenure.
- Analyzed retention rates across tenure groups.
- Evaluated contract-wise retention.
- Estimated monthly recurring revenue at risk from churned customers.

### 4. Customer Segmentation & Risk Analysis

Customers were segmented into:

- New Customers: Up to 12 months
- Regular Customers: 13–48 months
- Long-Term Customers: Above 48 months

A rule-based churn risk scoring approach was developed using:
- Monthly charges above the dataset median
- Month-to-month contract
- Tenure of 12 months or less

| Risk Category | Customers | Churned | Churn Rate |
|---|---:|---:|---:|
| Low Risk | 3,800 | 331 | 8.71% |
| Medium Risk | 2,431 | 970 | 39.90% |
| High Risk | 812 | 568 | 69.95% |

### 5. Machine Learning – Churn Prediction

Two classification models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier

Data preprocessing included:
- Standard Scaling for numerical features
- One-Hot Encoding for categorical features
- 80:20 Train-Test Split
- Stratified sampling

**Model Performance**

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.6% | 77.0% |
| Precision | 65.7% | 55.9% |
| Recall | 55.9% | 63.4% |
| F1 Score | 60.4% | 59.4% |
| ROC AUC | 84.2% | 82.3% |

**Key Observation:**

Logistic Regression achieved higher accuracy, precision, F1 Score, and ROC AUC, while Random Forest achieved higher recall.

The model selection depends on the business objective and the relative cost of missed churn cases versus unnecessary retention interventions.

## Key Business Insights

- Overall customer churn rate is **26.54%**.
- Customer retention rate is **73.46%**.
- High-risk customers show a churn rate of **69.95%**.
- Medium-risk customers account for the largest number of churned customers among the three risk categories.
- Monthly recurring revenue associated with churned customers is approximately **$139,130.85**.

## Interactive HTML Dashboard

An automated dashboard was developed using Python and Plotly.

**Dashboard Features:**
- Business overview and KPI cards
- Customer churn distribution
- Contract-wise churn analysis
- Internet Service-wise churn analysis
- Payment Method-wise churn analysis
- Tenure-based churn analysis
- Customer risk segmentation
- Machine learning model comparison
- Feature importance insights

**Dashboard File:**

`dashboard/Customer_Churn_Retention_Dashboard.html`

Open the HTML file in a web browser to explore the dashboard.

## Project Structure

```text
FUTURE_DS_02/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── cleaned_customer_churn.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_customer_retention_analysis.ipynb
│   ├── 05_customer_segmentation.ipynb
│   └── 06_churn_prediction_model.ipynb
│
├── reports/
│   ├── churn_risk_analysis.csv
│   ├── contract_churn_analysis.csv
│   ├── contract_retention_analysis.csv
│   ├── customer_churn_predictions.csv
│   ├── customer_segment_analysis.csv
│   ├── feature_importance.csv
│   ├── final_model_summary.csv
│   ├── model_comparison.csv
│   ├── payment_churn_analysis.csv
│   └── tenure_retention_analysis.csv
│
├── dashboard/
│   └── Customer_Churn_Retention_Dashboard.html
│
├── create_dashboard.py
├── .gitignore
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/vivekpandey11/FUTURE_DS_02.git
cd FUTURE_DS_02
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib plotly jupyter
```

### 3. Generate the Dashboard

```bash
python create_dashboard.py
```

The generated dashboard will be saved at:

```text
dashboard/Customer_Churn_Retention_Dashboard.html
```

Open the HTML file in your browser.

**Note:** The processed dataset is included in the repository. The raw dataset is excluded through `.gitignore`. To rerun the complete notebooks, place the original dataset in `data/raw/`.

## Project Information

**Internship:** Future Interns – Data Science & Analytics

**Task:** Task 2 – Customer Retention & Churn Analysis

**Repository:** [FUTURE_DS_02](https://github.com/vivekpandey11/FUTURE_DS_02)

---

*This project demonstrates an end-to-end analytics workflow, combining data analysis, customer segmentation, machine learning, and business-oriented dashboard development.*
