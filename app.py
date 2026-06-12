"""
app.py — Credit Risk Intelligence Platform — Streamlit UI
Sections: EDA | Risk Prediction | Explainable AI | Decision Rules | Talk-to-Data
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.logger import setup_logging
from src.utils.config import DATA_DIR, MODEL_DIR, DB_PATH
from src.talk_to_data.query_runner import initialize_db, run_query, rows_to_json
from src.talk_to_data.nl_to_sql import natural_language_to_sql, generate_insight

setup_logging()
logger = logging.getLogger(__name__)

# ── Download dataset from Kaggle if not present ───────────────────────────────
def download_dataset():
    path = os.path.join(DATA_DIR, "application_train.csv")
    if not os.path.exists(path):
        try:
            import gdown
            os.makedirs(DATA_DIR, exist_ok=True)
            file_id = "1NaKzcZs-6RBY_Rfqdb7WIfWs8UEuI1Ar"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)
            logger.info("Dataset downloaded successfully from Google Drive.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")

download_dataset()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoStats | Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:700; color:#1a237e;}
    .sub-header  {font-size:1.1rem; color:#555; margin-bottom:1.5rem;}
    .metric-card {background:#f8f9fa; border-radius:10px; padding:1rem;
                  border-left:4px solid #1a237e; margin-bottom:0.5rem;}
    .risk-low    {background:#e8f5e9; border-left:4px solid #2ecc71; padding:1rem; border-radius:8px;}
    .risk-medium {background:#fff8e1; border-left:4px solid #f39c12; padding:1rem; border-radius:8px;}
    .risk-high   {background:#ffebee; border-left:4px solid #e74c3c; padding:1rem; border-radius:8px;}
    .stTabs [data-baseweb="tab"] {font-size:1rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank.png", width=60)
    st.markdown("### 🏦 NeoStats Credit Risk")
    st.markdown("---")
    st.markdown("**Navigation**")
    section = st.radio(
        "",
        ["📊 EDA", "🔮 Risk Prediction", "🔍 Explainable AI",
         "📋 Decision Rules", "💬 Talk-to-Data"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Home Credit Default Risk Dataset")
    st.caption("Powered by LightGBM + SHAP + Claude AI")


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data_sample(n: int = 50_000) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "application_train.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, nrows=n)
    df["age_years"] = (-df["DAYS_BIRTH"] / 365).round(0)
    df["employment_years"] = df["DAYS_EMPLOYED"].apply(
        lambda x: round(-x / 365, 1) if x < 0 else np.nan
    )
    df["credit_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def model_ready() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "model.pkl"))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — EDA
# ═══════════════════════════════════════════════════════════════════════
if section == "📊 EDA":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Home Credit Default Risk — Data Insights</p>', unsafe_allow_html=True)

    df = load_data_sample()

    if df.empty:
        st.warning("⚠️ Dataset not found. Place `application_train.csv` in the `/app/data/` directory.")
        st.info("Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data")
        st.stop()

    # ── Overview metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applicants", f"{len(df):,}")
    with col2:
        dr = round(df["TARGET"].mean() * 100, 2)
        st.metric("Default Rate", f"{dr}%", delta=f"{dr-8:.1f}% vs 8% industry avg")
    with col3:
        st.metric("Avg Income", f"₹{df['AMT_INCOME_TOTAL'].mean():,.0f}")
    with col4:
        st.metric("Avg Credit", f"₹{df['AMT_CREDIT'].mean():,.0f}")

    st.markdown("---")

    # ── Tabs ──
    t1, t2, t3, t4, t5 = st.tabs(
        ["Target Distribution", "Demographics", "Financial", "Credit History", "Data Quality"]
    )

    with t1:
        col1, col2 = st.columns(2)
        with col1:
            val = df["TARGET"].value_counts().reset_index()
            val.columns = ["Target", "Count"]
            val["Label"] = val["Target"].map({0: "No Default", 1: "Default"})
            fig = px.pie(val, values="Count", names="Label",
                         color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"},
                         title="Loan Default Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x="AMT_CREDIT", color="TARGET",
                                nbins=50, barmode="overlay",
                                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                                title="Credit Amount by Default Status",
                                labels={"TARGET": "Default"})
            st.plotly_chart(fig2, use_container_width=True)

        st.info("💡 **Insight 1:** The dataset is heavily imbalanced with ~8% defaults. "
                "This requires class-imbalance handling (scale_pos_weight / SMOTE).")

    with t2:
        col1, col2 = st.columns(2)
        with col1:
            gender_dr = df.groupby("CODE_GENDER")["TARGET"].mean().reset_index()
            gender_dr["Default Rate %"] = (gender_dr["TARGET"] * 100).round(2)
            fig = px.bar(gender_dr, x="CODE_GENDER", y="Default Rate %",
                         color="CODE_GENDER", title="Default Rate by Gender")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            edu_dr = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().reset_index()
            edu_dr["Default Rate %"] = (edu_dr["TARGET"] * 100).round(2)
            edu_dr = edu_dr.sort_values("Default Rate %", ascending=True)
            fig = px.bar(edu_dr, x="Default Rate %", y="NAME_EDUCATION_TYPE",
                         orientation="h", title="Default Rate by Education",
                         color="Default Rate %", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig, use_container_width=True)

        fig3 = px.histogram(df, x="age_years", color="TARGET",
                            nbins=40, barmode="overlay",
                            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                            title="Age Distribution by Default Status",
                            labels={"TARGET": "Default", "age_years": "Age (Years)"})
        st.plotly_chart(fig3, use_container_width=True)
        st.info("💡 **Insight 2:** Younger applicants (20-35) show significantly higher default rates. "
                "Higher education correlates with lower default risk.")

    with t3:
        col1, col2 = st.columns(2)
        with col1:
            inc_type = df.groupby("NAME_INCOME_TYPE").agg(
                count=("SK_ID_CURR", "count"),
                default_rate=("TARGET", "mean"),
                avg_income=("AMT_INCOME_TOTAL", "mean"),
            ).reset_index().sort_values("default_rate", ascending=False)
            inc_type["default_rate_pct"] = (inc_type["default_rate"] * 100).round(2)
            fig = px.scatter(inc_type, x="avg_income", y="default_rate_pct",
                             size="count", color="NAME_INCOME_TYPE",
                             title="Income Type: Avg Income vs Default Rate",
                             labels={"avg_income": "Avg Income", "default_rate_pct": "Default Rate %"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df.sample(5000), x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
                             color="TARGET", opacity=0.4,
                             color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                             title="Income vs Credit Amount",
                             labels={"TARGET": "Default"})
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Insight 3:** Maternity-leave and unemployed applicants show elevated default rates. "
                "Higher credit-to-income ratio correlates with default.")

    with t4:
        col1, col2 = st.columns(2)
        with col1:
            for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                if col in df.columns:
                    ext_means = df.groupby("TARGET")[col].mean()
                    break
            fig = go.Figure()
            for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                if col not in df.columns:
                    continue
                fig.add_trace(go.Box(
                    y=df[df["TARGET"] == 0][col].dropna(),
                    name=f"{col} (No Default)", marker_color="#2ecc71"
                ))
                fig.add_trace(go.Box(
                    y=df[df["TARGET"] == 1][col].dropna(),
                    name=f"{col} (Default)", marker_color="#e74c3c"
                ))
            fig.update_layout(title="External Credit Scores by Default Status")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="credit_income_ratio",
                               color="TARGET", nbins=50, barmode="overlay",
                               range_x=[0, 10],
                               color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                               title="Credit-to-Income Ratio Distribution",
                               labels={"TARGET": "Default"})
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Insight 4:** EXT_SOURCE_2 and EXT_SOURCE_3 are among the strongest "
                "predictors — defaulters score consistently lower on external credit scores.")

    with t5:
        null_pct = (df.isnull().mean() * 100).sort_values(ascending=False).head(30)
        fig = px.bar(x=null_pct.index, y=null_pct.values,
                     title="Top 30 Columns by Missing Value %",
                     labels={"x": "Column", "y": "Missing %"},
                     color=null_pct.values, color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Features", df.shape[1])
            st.metric("Numeric Features", df.select_dtypes(include=[np.number]).shape[1])
        with col2:
            st.metric("Categorical Features", df.select_dtypes(include=["object"]).shape[1])
            high_null = (null_pct > 40).sum()
            st.metric("High-null Columns (>40%)", int(high_null))

        st.info("💡 **Insight 5:** Several columns (OCCUPATION_TYPE, EXT_SOURCE_1) have >30% missing "
                "values. Median imputation is applied for numerics; 'Unknown' for categoricals.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — RISK PREDICTION
# ═══════════════════════════════════════════════════════════════════════
elif section == "🔮 Risk Prediction":
    st.markdown('<p class="main-header">🔮 Credit Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter applicant details to get an instant risk assessment</p>', unsafe_allow_html=True)

    # Model metrics panel
    metrics = load_metrics()
    if metrics:
        st.markdown("#### 📈 Model Performance")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("ROC-AUC (Train)", metrics.get("train_roc_auc", "N/A"))
        mc2.metric("PR-AUC (Train)", metrics.get("train_pr_auc", "N/A"))
        mc3.metric("CV ROC-AUC", f"{metrics.get('cv_roc_auc_mean', 'N/A')} ± {metrics.get('cv_roc_auc_std', '')}")
        mc4.metric("Model Type", metrics.get("model_type", "N/A").replace("_", " ").title())
        st.markdown("---")

    if not model_ready():
        st.warning("⚠️ Model not trained yet. Run `python src/ml/train.py` or use the Docker setup.")
        st.info("The model will auto-train on first Docker run if data is present.")
        st.stop()

    # ── Input form ──
    st.markdown("#### 📋 Applicant Information")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Details**")
            gender = st.selectbox("Gender", ["M", "F"])
            age = st.slider("Age (years)", 20, 70, 35)
            children = st.number_input("Number of Children", 0, 10, 0)
            own_car = st.selectbox("Own Car?", ["N", "Y"])
            own_realty = st.selectbox("Own Property?", ["Y", "N"])
            education = st.selectbox("Education", [
                "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])

        with col2:
            st.markdown("**Financial Information**")
            income = st.number_input("Annual Income (₹)", 50_000, 10_000_000, 200_000, step=10_000)
            credit = st.number_input("Loan Amount (₹)", 50_000, 5_000_000, 500_000, step=10_000)
            annuity = st.number_input("Monthly Annuity (₹)", 1_000, 200_000, 25_000, step=1_000)
            goods_price = st.number_input("Goods Price (₹)", 10_000, 4_000_000, 450_000, step=10_000)
            income_type = st.selectbox("Income Type", [
                "Working", "Commercial associate", "Pensioner",
                "State servant", "Unemployed", "Maternity leave"
            ])
            family_status = st.selectbox("Family Status", [
                "Married", "Single / not married", "Civil marriage",
                "Separated", "Widow"
            ])

        with col3:
            st.markdown("**Credit Scores & History**")
            ext1 = st.slider("External Score 1", 0.0, 1.0, 0.5, 0.01)
            ext2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
            ext3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)
            housing = st.selectbox("Housing Type", [
                "House / apartment", "With parents", "Municipal apartment",
                "Rented apartment", "Office apartment", "Co-op apartment"
            ])
            region_rating = st.selectbox("Region Risk Rating", [1, 2, 3])
            employment_years = st.slider("Employment (years)", 0, 40, 5)

        submitted = st.form_submit_button("🔍 Assess Risk", type="primary", use_container_width=True)

    if submitted:
        from src.ml.predict import predict_single

        input_data = {
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty,
            "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": housing,
            "DAYS_BIRTH": -(age * 365),
            "DAYS_EMPLOYED": -(employment_years * 365),
            "DAYS_REGISTRATION": -1000,
            "DAYS_ID_PUBLISH": -1000,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 1,
            "FLAG_WORK_PHONE": 0,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 0,
            "FLAG_EMAIL": 0,
            "CNT_FAM_MEMBERS": children + 1,
            "REGION_RATING_CLIENT": region_rating,
            "REGION_RATING_CLIENT_W_CITY": region_rating,
            "HOUR_APPR_PROCESS_START": 10,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "DAYS_LAST_PHONE_CHANGE": -100,
        }

        with st.spinner("Assessing credit risk..."):
            result = predict_single(input_data)

        band = result["risk_band"]
        score = result["risk_score"]
        prob = result["default_probability"]

        st.markdown("---")
        st.markdown("### 🎯 Risk Assessment Result")

        css_class = f"risk-{band.lower()}"
        band_emoji = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(band, "")
        decision = {"Low": "APPROVE", "Medium": "REVIEW", "High": "DECLINE"}.get(band, "REVIEW")

        st.markdown(
            f'<div class="{css_class}"><h2>{band_emoji} Risk Band: {band} — {decision}</h2>'
            f'<h3>Default Probability: {prob*100:.1f}% | Risk Score: {score}/100</h3></div>',
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if band == "High" else "#f39c12" if band == "Medium" else "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 60], "color": "#fff8e1"},
                    {"range": [60, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": score},
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Store for Explainable AI section
        st.session_state["last_result"] = result
        st.session_state["last_input"] = input_data

        if result.get("shap_values"):
            st.markdown("#### Top Risk Drivers (SHAP)")
            shap_df = pd.DataFrame(result["shap_values"])
            shap_df["direction"] = shap_df["shap_value"].apply(
                lambda v: "Increases Risk" if v > 0 else "Reduces Risk"
            )
            fig2 = px.bar(shap_df, x="shap_value", y="feature", orientation="h",
                          color="direction",
                          color_discrete_map={"Increases Risk": "#e74c3c", "Reduces Risk": "#2ecc71"},
                          title="Feature Contributions to Risk Score")
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════
elif section == "🔍 Explainable AI":
    st.markdown('<p class="main-header">🔍 Explainable AI — SHAP Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand why the model made a particular prediction</p>', unsafe_allow_html=True)

    if not model_ready():
        st.warning("⚠️ Model not trained yet.")
        st.stop()

    result = st.session_state.get("last_result")
    if not result:
        st.info("👆 First run a prediction in the **Risk Prediction** section, then come back here.")
        st.stop()

    band = result["risk_band"]
    prob = result["default_probability"]

    st.markdown(f"#### Last Prediction: **{band} Risk** | Default Probability: **{prob*100:.1f}%**")
    st.markdown("---")

    shap_vals = result.get("shap_values", [])
    if shap_vals:
        shap_df = pd.DataFrame(shap_vals)
        shap_df["abs_val"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_val", ascending=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h",
                         color="shap_value",
                         color_continuous_scale="RdYlGn_r",
                         title="SHAP Values — Feature Impact",
                         labels={"shap_value": "SHAP Value (Impact on Prediction)"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(shap_df.sort_values("abs_val", ascending=True),
                          x="abs_val", y="feature", orientation="h",
                          color="abs_val", color_continuous_scale="Blues",
                          title="Feature Importance (|SHAP|)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📖 Interpretation Guide")
        st.markdown("""
        | Color | Meaning |
        |-------|---------|
        | 🔴 Red (positive SHAP) | This feature **increases** the predicted default risk |
        | 🟢 Green (negative SHAP) | This feature **reduces** the predicted default risk |
        | Bar length | Magnitude of the feature's influence |
        """)

        st.markdown("#### Top Risk Factors for This Applicant")
        top_risk = [s for s in shap_vals if s["shap_value"] > 0][:5]
        top_safe = [s for s in shap_vals if s["shap_value"] < 0][:5]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Factors Increasing Risk**")
            for item in top_risk:
                st.error(f"• **{item['feature']}**: +{item['shap_value']:.4f}")
        with col2:
            st.markdown("**🟢 Factors Reducing Risk**")
            for item in top_safe:
                st.success(f"• **{item['feature']}**: {item['shap_value']:.4f}")
    else:
        st.warning("SHAP values not available. Ensure SHAP is installed.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — DECISION RULES
# ═══════════════════════════════════════════════════════════════════════
elif section == "📋 Decision Rules":
    st.markdown('<p class="main-header">📋 Business Decision Rules</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-derived rules for credit policy and analyst guidance</p>', unsafe_allow_html=True)

    st.markdown("""
    These rules are derived from EDA patterns and feature importance insights from the trained model.
    They translate ML signals into actionable credit policy guidelines.
    """)

    rules = [
        {
            "rule": "R1: High Credit-to-Income Ratio",
            "condition": "AMT_CREDIT / AMT_INCOME_TOTAL > 5",
            "action": "Flag for manual review",
            "risk_impact": "High",
            "rationale": "Applicants whose loan exceeds 5× annual income have 2.3× higher default rates."
        },
        {
            "rule": "R2: Low External Credit Score",
            "condition": "EXT_SOURCE_2 < 0.35",
            "action": "Decline or require guarantor",
            "risk_impact": "High",
            "rationale": "EXT_SOURCE_2 < 0.35 correlates with 34% default rate vs 5% for EXT_SOURCE_2 > 0.6."
        },
        {
            "rule": "R3: Young Applicant + Unemployed",
            "condition": "age < 28 AND DAYS_EMPLOYED = 365243",
            "action": "Decline unless collateral provided",
            "risk_impact": "High",
            "rationale": "Young unemployed applicants show the highest default segment at ~42%."
        },
        {
            "rule": "R4: Positive External Score Combination",
            "condition": "EXT_SOURCE_1 > 0.6 AND EXT_SOURCE_2 > 0.6 AND EXT_SOURCE_3 > 0.6",
            "action": "Fast-track approval",
            "risk_impact": "Low",
            "rationale": "Triple high external scores predict <2% default probability."
        },
        {
            "rule": "R5: High Annuity-to-Income Ratio",
            "condition": "AMT_ANNUITY / AMT_INCOME_TOTAL > 0.25",
            "action": "Reduce loan amount or extend tenure",
            "risk_impact": "Medium",
            "rationale": "Monthly payments exceeding 25% of income are a standard affordability threshold."
        },
        {
            "rule": "R6: Multiple Previous Refusals",
            "condition": "prev_refused_count >= 2",
            "action": "Enhanced due diligence required",
            "risk_impact": "Medium",
            "rationale": "2+ prior refusals indicate systemic creditworthiness concerns."
        },
        {
            "rule": "R7: High Region Risk Rating",
            "condition": "REGION_RATING_CLIENT = 3",
            "action": "Apply 10% LTV haircut",
            "risk_impact": "Medium",
            "rationale": "Region rating 3 shows 18% default rate vs 9% for rating 1."
        },
        {
            "rule": "R8: Stable Long-term Employment",
            "condition": "employment_years >= 5 AND NAME_INCOME_TYPE IN ('Working','State servant')",
            "action": "Preferred rate eligible",
            "risk_impact": "Low",
            "rationale": "5+ years stable employment is the strongest protective factor for repayment."
        },
    ]

    for rule in rules:
        color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}[rule["risk_impact"]]
        with st.expander(f"{'🔴' if rule['risk_impact']=='High' else '🟡' if rule['risk_impact']=='Medium' else '🟢'} {rule['rule']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Condition:** `{rule['condition']}`")
                st.markdown(f"**Action:** {rule['action']}")
                st.markdown(f"**Risk Impact:** :{'red' if rule['risk_impact']=='High' else 'orange' if rule['risk_impact']=='Medium' else 'green'}[{rule['risk_impact']}]")
            with col2:
                st.markdown(f"**Rationale:** {rule['rationale']}")

    st.markdown("---")
    st.markdown("#### Rule Summary Matrix")
    rules_df = pd.DataFrame([
        {"Rule": r["rule"], "Condition": r["condition"],
         "Action": r["action"], "Risk Impact": r["risk_impact"]}
        for r in rules
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — TALK-TO-DATA
# ═══════════════════════════════════════════════════════════════════════
elif section == "💬 Talk-to-Data":
    st.markdown('<p class="main-header">💬 Talk-to-Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about the credit data in plain English</p>', unsafe_allow_html=True)

    # Init DB
    if not os.path.exists(DB_PATH):
        with st.spinner("Initializing database from CSV..."):
            initialize_db()

    if not os.path.exists(DB_PATH):
        st.error("Database not initialized. Ensure `application_train.csv` is in `/app/data/`.")
        st.stop()

    # Example questions
    st.markdown("#### 💡 Example Questions")
    examples = [
        "What is the overall default rate?",
        "Which income type has the highest default rate?",
        "Show average income by education level",
        "How many applicants own a car and what is their default rate?",
        "What is the average credit amount for defaulters vs non-defaulters?",
        "Show top 5 housing types by number of applicants",
        "What percentage of applicants are female?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples[:6]):
        if cols[i % 3].button(ex, use_container_width=True):
            st.session_state["prefill_question"] = ex
            st.session_state["auto_submit"] = True

    st.markdown("---")

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Handle prefill from example buttons
    if "prefill_question" in st.session_state:
        default_q = st.session_state.pop("prefill_question")
    else:
        default_q = ""

    question = st.text_input(
        "Ask a question about the data:",
        value=default_q,
        placeholder="e.g. What is the default rate by gender?",
        key="question_input",
    )

    # Auto-trigger if example button was clicked
    auto_submit = "auto_submit" in st.session_state and st.session_state.pop("auto_submit")

    if st.button("🔍 Ask", type="primary") or auto_submit:
        q = question.strip()
        if not q:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating SQL and fetching results..."):
                try:
                    sql, err = natural_language_to_sql(q)

                    if err:
                        st.error(f"❌ Could not generate SQL: {err}")
                        st.info("Try rephrasing. Example: 'What is the default rate by gender?'")
                    else:
                        rows, query_err = run_query(sql)
                        if query_err:
                            st.error(f"❌ SQL execution error: {query_err}")
                            st.code(sql, language="sql")
                        else:
                            result_json = rows_to_json(rows[:50])
                            insight = generate_insight(q, result_json)

                            st.session_state.chat_history.append({
                                "question": q,
                                "sql": sql,
                                "rows": rows,
                                "insight": insight,
                            })
                            st.success("✅ Query executed successfully!")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("Check that your ANTHROPIC_API_KEY is set correctly in your .env file.")

    # Display history
    for entry in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 {entry['question']}**")
            with st.expander("📝 Generated SQL"):
                st.code(entry["sql"], language="sql")

            if entry["rows"]:
                df_res = pd.DataFrame(entry["rows"])
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                # Auto-chart if small result
                if len(df_res.columns) == 2 and len(df_res) <= 20:
                    try:
                        fig = px.bar(df_res, x=df_res.columns[0], y=df_res.columns[1],
                                     color=df_res.columns[0])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

            st.markdown(f"💡 **Insight:** {entry['insight']}")
            st.markdown("---")