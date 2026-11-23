
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from html import escape

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "cleaned_customer_churn.csv"
REPORTS = BASE / "reports"
OUTPUT_DIR = BASE / "dashboard"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load and prepare data
df = pd.read_csv(DATA)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

df["Tenure Group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
)

total = len(df)
churned = (df["Churn"] == "Yes").sum()
retained = (df["Churn"] == "No").sum()
churn_rate = churned / total * 100 if total else 0
retention_rate = retained / total * 100 if total else 0
revenue_risk = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
avg_monthly = df["MonthlyCharges"].mean()

# Rule-based risk segmentation, matching the project analysis
monthly_median = df["MonthlyCharges"].median()
df["Risk Score"] = (
    (df["MonthlyCharges"] > monthly_median).astype(int)
    + (df["Contract"] == "Month-to-month").astype(int)
    + (df["tenure"] <= 12).astype(int)
)

df["Risk Category"] = pd.cut(
    df["Risk Score"],
    bins=[-1, 1, 2, 3],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# 2. Chart helper
charts = []

def add_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", color="#243247"),
        margin=dict(l=35, r=25, t=65, b=45),
        title_font=dict(size=17, color="#172B4D"),
        legend_title_text=""
    )
    charts.append(fig.to_html(
        full_html=False,
        include_plotlyjs=True if not charts else False
    ))

# 3. Churn distribution
churn_counts = df["Churn"].value_counts().reset_index()
churn_counts.columns = ["Churn Status", "Customers"]

fig = px.pie(
    churn_counts,
    names="Churn Status",
    values="Customers",
    hole=0.58,
    title="Customer Churn Distribution",
    color="Churn Status",
    color_discrete_map={"Yes": "#E45756", "No": "#20A39E"}
)
add_chart(fig)

# 4. Churn by category
def churn_by_category(column, title):
    temp = df.groupby(column, dropna=False)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index(name="Churn Rate")

    temp[column] = temp[column].astype(str)

    fig = px.bar(
        temp,
        x=column,
        y="Churn Rate",
        title=title,
        text_auto=".1f",
        color="Churn Rate",
        color_continuous_scale="Blues"
    )
    fig.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="")
    add_chart(fig)

churn_by_category("Contract", "Churn Rate by Contract")
churn_by_category("InternetService", "Churn Rate by Internet Service")
churn_by_category("PaymentMethod", "Churn Rate by Payment Method")
churn_by_category("Tenure Group", "Churn Rate by Tenure Group")

# 5. Average monthly charges by churn status
charges = df.groupby("Churn")["MonthlyCharges"].mean().reset_index()

fig = px.bar(
    charges,
    x="Churn",
    y="MonthlyCharges",
    title="Average Monthly Charges by Churn Status",
    text_auto=".2f",
    color="Churn",
    color_discrete_map={"Yes": "#E45756", "No": "#20A39E"}
)
fig.update_layout(xaxis_title="Churn (Yes / No)", yaxis_title="Average Monthly Charges")
add_chart(fig)

# 6. Customer risk analysis
risk = df.groupby("Risk Category", observed=False).agg(
    Total_Customers=("Churn", "size"),
    Churned_Customers=("Churn", lambda x: (x == "Yes").sum())
).reset_index()

risk["Churn Rate"] = (
    risk["Churned_Customers"] / risk["Total_Customers"] * 100
)

fig = px.bar(
    risk,
    x="Risk Category",
    y="Churn Rate",
    title="Churn Rate Across Customer Risk Categories",
    text_auto=".1f",
    color="Risk Category",
    color_discrete_map={
        "Low Risk": "#20A39E",
        "Medium Risk": "#F4B942",
        "High Risk": "#E45756"
    }
)
fig.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="")
add_chart(fig)

# 7. Model comparison
model_path = REPORTS / "model_comparison.csv"
model_html = "<p>Model comparison file was not found.</p>"

if model_path.exists():
    models = pd.read_csv(model_path)
    model_html = models.to_html(
        index=False,
        classes="data-table",
        border=0,
        float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    )

    # Build a comparison chart if expected metric columns are present
    normalized = {
        str(c).strip().lower().replace("_", " ").replace("-", " "): c
        for c in models.columns
    }

    model_col = next(
        (v for k, v in normalized.items() if k == "model"), None
    )

    metric_names = ["accuracy", "precision", "recall", "f1 score", "roc auc"]
    metric_cols = {
        name: normalized[name]
        for name in metric_names
        if name in normalized
    }

    if model_col and metric_cols:
        plot_data = models.melt(
            id_vars=[model_col],
            value_vars=list(metric_cols.values()),
            var_name="Metric",
            value_name="Score"
        )
        reverse = {v: k.title() for k, v in metric_cols.items()}
        plot_data["Metric"] = plot_data["Metric"].map(reverse)

        fig = px.bar(
            plot_data,
            x="Metric",
            y="Score",
            color=model_col,
            barmode="group",
            title="Machine Learning Model Performance",
            text_auto=".3f"
        )
        fig.update_layout(yaxis_title="Score", xaxis_title="")
        add_chart(fig)

# 8. Optional feature importance table
feature_path = REPORTS / "feature_importance.csv"
feature_html = "<p>Feature importance report not available.</p>"

if feature_path.exists():
    features = pd.read_csv(feature_path)
    feature_html = features.head(15).to_html(
        index=False, classes="data-table", border=0
    )

# 9. Optional prediction report information
prediction_path = REPORTS / "customer_churn_predictions.csv"
prediction_note = "Prediction report not found."

if prediction_path.exists():
    predictions = pd.read_csv(prediction_path)
    prediction_note = (
        f"Prediction records available: {len(predictions):,}. "
        "These are test-set predictions, not predictions for the entire customer base."
    )

# 10. Assemble dashboard
chart_sections = ""
for i, chart in enumerate(charts):
    chart_sections += f'<section class="chart">{chart}</section>\n'

cards = [
    ("Total Customers", f"{total:,}"),
    ("Churned Customers", f"{churned:,}"),
    ("Retained Customers", f"{retained:,}"),
    ("Churn Rate", f"{churn_rate:.2f}%"),
    ("Retention Rate", f"{retention_rate:.2f}%"),
    ("Avg. Monthly Charges", f"${avg_monthly:,.2f}"),
    ("Monthly Revenue at Risk", f"${revenue_risk:,.2f}")
]

cards_html = "".join(
    f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
    f'<div class="kpi-value">{escape(value)}</div></div>'
    for label, value in cards
)

html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Customer Churn & Retention Dashboard</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    background: #F4F6FA;
    color: #243247;
    font-family: Arial, sans-serif;
}}
header {{
    background: #172B4D;
    color: white;
    padding: 30px 5%;
}}
header h1 {{ margin: 0 0 8px; font-size: 28px; }}
header p {{ margin: 0; color: #DCE5F2; }}
main {{ max-width: 1450px; margin: auto; padding: 25px 4%; }}
h2 {{ color: #172B4D; margin-top: 30px; }}
.kpis {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 20px 0 30px;
}}
.kpi {{
    background: white;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(23,43,77,.07);
    border-left: 4px solid #3B82F6;
}}
.kpi-label {{ color: #65758B; font-size: 13px; margin-bottom: 10px; }}
.kpi-value {{ font-size: 25px; font-weight: bold; color: #172B4D; }}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 18px;
}}
.chart {{
    background: white;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(23,43,77,.06);
    min-width: 0;
}}
.data-table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    margin: 15px 0 25px;
}}
.data-table th, .data-table td {{
    border: 1px solid #E5EAF1;
    padding: 12px;
    text-align: left;
}}
.data-table th {{ background: #172B4D; color: white; }}
.note {{
    background: #EAF1FB;
    border-left: 4px solid #3B82F6;
    padding: 15px;
    border-radius: 6px;
    line-height: 1.6;
}}
footer {{
    text-align: center;
    color: #65758B;
    padding: 25px;
    font-size: 13px;
}}
@media(max-width:600px) {{
    .grid {{ grid-template-columns: 1fr; }}
    header h1 {{ font-size: 22px; }}
    .kpi-value {{ font-size: 21px; }}
}}
</style>
</head>
<body>
<header>
    <h1>Customer Churn & Retention Dashboard</h1>
    <p>Customer Analytics | Retention Insights | Risk Segmentation | ML Model Evaluation</p>
</header>
<main>
    <h2>Business Overview</h2>
    <div class="kpis">{cards_html}</div>

    <h2>Churn & Retention Analysis</h2>
    <div class="grid">{chart_sections}</div>

    <h2>Machine Learning Model Comparison</h2>
    {model_html}

    <h2>Feature Importance</h2>
    {feature_html}

    <h2>Methodology & Notes</h2>
    <div class="note">
        <b>Customer Risk:</b> Rule-based segmentation using monthly charges,
        month-to-month contract, and tenure. It is not an ML prediction.<br>
        <b>Monthly Revenue at Risk:</b> Sum of monthly charges for customers
        marked as churned. This is an estimate of recurring revenue at risk,
        not profit or a verified accounting loss.<br>
        <b>Model Predictions:</b> {escape(prediction_note)}<br>
        <b>Churn Rate:</b> Churned customers divided by total customers.
    </div>
</main>
<footer>Customer Churn & Retention Analytics | Generated with Python and Plotly</footer>
</body>
</html>
"""

output_file = OUTPUT_DIR / "Customer_Churn_Retention_Dashboard.html"
output_file.write_text(html, encoding="utf-8")

print("\nDashboard generated successfully!")
print(f"Saved at: {output_file}")
print(f"Total customers: {total:,}")
print(f"Churn rate: {churn_rate:.2f}%")
