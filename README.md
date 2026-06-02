# 🏦 NeoStats — AI-Powered Credit Risk Intelligence Platform

> **Intelligence. Innovation. Impact.**  
> A full-stack AI platform for credit default prediction, explainability, and conversational data analysis.

---

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Quick Start (Docker)](#quick-start-docker)
3. [Dataset Setup](#dataset-setup)
4. [Module Descriptions](#module-descriptions)
5. [Model Selection & Imbalance Strategy](#model-selection--class-imbalance-strategy)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Talk-to-Data: Prompt Engineering](#talk-to-data-prompt-engineering)
8. [Decision Rules Logic](#decision-rules-logic)
9. [Known Limitations & Improvements](#known-limitations--improvements)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────────┐  │
│  │   EDA    │ │  Risk    │ │ Explain- │ │Rules │ │Talk-to-  │  │
│  │Dashboard │ │Prediction│ │  able AI │ │      │ │  Data    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────┘ └──────────┘  │
└─────────────────┬───────────────┬──────────────┬────────────────┘
                  │               │              │
         ┌────────▼──────┐  ┌────▼────┐  ┌──────▼──────────┐
         │  ML Pipeline  │  │  SHAP   │  │  NL→SQL Agent   │
         │  (LightGBM)   │  │Explainer│  │ (Groq LLaMA 3.3)│
         └────────┬──────┘  └─────────┘  └──────┬──────────┘
                  │                              │
         ┌────────▼──────┐              ┌────────▼──────────┐
         │ Data Pipeline  │              │  SQLite Database  │
         │loader+preproc  │              │ (from CSV on init)│
         └───────────────┘              └───────────────────┘
```

### Component Map

| File | Role |
|------|------|
| `app.py` | Streamlit multi-section UI entry point |
| `src/data/loader.py` | Load & join all Home Credit tables |
| `src/data/preprocessor.py` | Feature engineering, encoding, imputation |
| `src/ml/train.py` | LightGBM training pipeline |
| `src/ml/predict.py` | Inference + SHAP explanation |
| `src/ml/evaluate.py` | Metrics: ROC-AUC, PR-AUC, classification report |
| `src/talk_to_data/nl_to_sql.py` | NL→SQL via Groq API (LLaMA 3.3 70B) |
| `src/talk_to_data/query_runner.py` | SQLite execution layer |
| `src/talk_to_data/prompt_templates.py` | Versioned prompts + schema |
| `src/utils/` | Logging, config, helpers |
| `notebooks/eda.py` | Standalone EDA script |

---

## Quick Start (Docker)

### Prerequisites
- Docker & Docker Compose installed
- Groq API key (free at https://console.groq.com)
- Home Credit dataset CSVs (see below)

### Steps

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd credit_risk_platform

# 2. Set up environment
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY=your-key-here

# 3. Place dataset files in ./data/
#    (at minimum: application_train.csv)
ls data/application_train.csv

# 4. Build and run
docker compose up --build

# 5. Open browser
open http://localhost:8501
```

The container will:
- Auto-train the model if `model.pkl` does not exist and data is present
- Initialize the SQLite DB from the CSV
- Launch the Streamlit UI

### Manual Run (without Docker)

```bash
pip install -r requirements.txt

# Set environment variables (Windows PowerShell)
$env:DATA_DIR = "data"
$env:MODEL_DIR = "models"
$env:GROQ_API_KEY = "your-groq-key-here"

# Or create a .env file with those values and run:
streamlit run app.py
```

---

## Dataset Setup

Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data

Place the following files in the `./data/` directory:

| File | Required | Description |
|------|----------|-------------|
| `application_train.csv` | **Required** | Main training table |
| `bureau.csv` | Optional | Credit bureau history |
| `previous_application.csv` | Optional | Previous loan applications |
| `installments_payments.csv` | Optional | Payment history |

The platform degrades gracefully if optional files are missing — the model uses only `application_train.csv` columns.

---

## Module Descriptions

### 1. Data Pipeline (`src/data/`)

**loader.py**
- Loads `application_train.csv` and optionally joins bureau, previous applications, and installment data
- Each auxiliary table is aggregated per applicant (`SK_ID_CURR`) before joining

**preprocessor.py**
- Engineered features: `credit_income_ratio`, `annuity_income_ratio`, `credit_goods_ratio`, `age_years`, `employment_years`
- Handles anomalous `DAYS_EMPLOYED = 365243` (unemployed placeholder → NaN)
- Label-encodes categorical columns with `LabelEncoder`
- Median imputation for all numeric features

### 2. ML Layer (`src/ml/`)

See [Model Selection](#model-selection--class-imbalance-strategy) below.

### 3. Talk-to-Data (`src/talk_to_data/`)

See [Prompt Engineering](#talk-to-data-prompt-engineering) below.

---

## Model Selection & Class Imbalance Strategy

### Model: LightGBM

**Why LightGBM?**
- Natively handles missing values (no imputation strictly required, though we apply it for robustness)
- Histogram-based training: extremely fast on 300k+ row datasets
- Built-in feature importance used as SHAP fallback
- Superior to logistic regression on non-linear credit data
- Better than XGBoost for speed at equivalent accuracy

**Fallback**: If LightGBM is unavailable, the pipeline falls back to `RandomForestClassifier(class_weight='balanced')`.

### Class Imbalance Strategy

The dataset is heavily imbalanced (~8% default rate). We use two complementary techniques:

1. **`scale_pos_weight`** — LightGBM parameter set to `n_negative / n_positive` (≈11.5). This penalises false negatives more, increasing recall for the minority class.

2. **Stratified K-Fold CV** (5 folds) — Ensures each validation fold preserves the class ratio, giving a reliable AUC estimate.

3. **PR-AUC as secondary metric** — For imbalanced datasets, PR-AUC is more informative than ROC-AUC alone. Both are reported.

**Why not SMOTE?** SMOTE can introduce noise in high-dimensional tabular data. `scale_pos_weight` achieves similar recall improvement without synthetic data artifacts.

---

## Evaluation Metrics

| Metric | Typical Range | Notes |
|--------|--------------|-------|
| ROC-AUC | 0.75–0.80 | Primary metric |
| PR-AUC | 0.35–0.45 | More informative for imbalanced data |
| CV ROC-AUC | 0.74–0.79 | 5-fold stratified cross-validation |

Metrics are saved to `models/metrics.json` and displayed in the UI.

Industry benchmark: Kaggle leaderboard top solutions achieve ~0.80 ROC-AUC with heavy feature engineering across all 7 tables. Our lightweight pipeline targets ~0.75-0.77 ROC-AUC.

---

## Talk-to-Data: Prompt Engineering

### LLM Choice: Groq (LLaMA 3.3 70B Versatile)

**Why Groq instead of Claude?**
- Free tier with no credit card required — ideal for open evaluation
- LLaMA 3.3 70B achieves strong NL→SQL accuracy comparable to GPT-4 class models
- Groq's inference hardware delivers very low latency (<1s for SQL generation)
- Same OpenAI-compatible API format — easy to swap LLMs if needed

### Architecture
```
User Question → Groq API (LLaMA 3.3 70B) → SQL Validation → SQLite → Result → Groq API (Insight) → UI
```

### Prompt Design

**NL→SQL System Prompt (`prompt_templates.py`)**
- Provides complete schema with column descriptions and data type context
- Explicit rules prevent hallucination:
  - "Return ONLY valid SQLite SQL — no markdown, no explanation"
  - Maps domain language ("defaulters", "bad customers") to filter conditions
  - Handles DAYS_BIRTH sign convention explicitly
  - Instructs model to return an error SELECT when the question is unanswerable
  - Explicitly forbids write operations (DDL/DML guard)

**Insight Generation Prompt**
- Separate, focused prompt for converting SQL results to plain English
- "Do not fabricate data not in the result" — hallucination guard
- Max 300 tokens — keeps responses concise

**SQL Validation Layer (`nl_to_sql.py`)**
- Regex check for forbidden keywords: DROP, DELETE, UPDATE, INSERT, ALTER
- Verifies query contains SELECT and references a known table
- Strips markdown fences from LLM output

**Token Optimization**
- Schema is included once in system prompt (not per-turn)
- Max tokens: 512 for SQL, 300 for insight
- Stateless per query — no conversation history maintained (cheaper, faster)

### Example Queries (5 Verified Working)

| Question | Result |
|----------|--------|
| What is the overall default rate? | Returns default % across all applicants |
| Which income type has the highest default rate? | Ranked list by income type |
| Show average income by education level | Average income per education category |
| How many applicants own a car and what is their default rate? | Car vs no-car comparison |
| What is the average credit amount for defaulters vs non-defaulters? | Side-by-side comparison |

---

## Decision Rules Logic

Rules are derived from three sources:

1. **EDA patterns** — e.g. default rate by age band, income type
2. **SHAP feature importance** — top features from the trained model
3. **Industry credit policy standards** — 25% annuity-to-income threshold, LTV haircuts

Rules are structured as: `Condition → Action → Risk Impact → Rationale`

See the **Decision Rules** section in the UI for the full rule set with explanations.

---

## Known Limitations & Improvements

### Current Limitations

| Area | Limitation |
|------|-----------|
| Data | Only `application_train.csv` is strictly required; more tables improve accuracy |
| Model | Single model; no ensemble or stacking |
| SHAP | TreeExplainer can be slow for large batches |
| NL→SQL | Limited to single-table queries on `applications` |
| UI | No user authentication or multi-user session isolation |

### Potential Improvements

1. **Feature engineering** — Add all 7 dataset tables with richer aggregations
2. **Model ensemble** — Blend LightGBM + CatBoost + Logistic Regression
3. **SHAP caching** — Cache explainer per model artifact for faster inference
4. **Multi-table SQL** — Extend SQLite schema to include bureau/installments tables
5. **Model monitoring** — Track prediction drift over time
6. **A/B testing** — Evaluate rule changes against model predictions
7. **Async training** — Background model retraining without blocking UI

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| UI | Streamlit | Rapid prototyping, data-centric apps |
| ML | LightGBM | Speed + accuracy + missing value handling |
| Explainability | SHAP TreeExplainer | Industry standard for tree models |
| LLM | Groq (LLaMA 3.3 70B) | Free tier, low latency, strong NL→SQL accuracy |
| Database | SQLite | Zero-config, embedded, portable |
| Visualization | Plotly | Interactive charts in Streamlit |
| Deployment | Docker + Docker Compose | Single-command reproducibility |

---

## File Structure

```
credit_risk_platform/
├── data/                          ← Home Credit dataset files (not committed)
├── documents/
│   └── project_presentation.pdf  ← Project presentation
├── models/                        ← Saved model artifacts (not committed)
├── notebooks/
│   ├── eda.py                     ← Standalone EDA script
│   └── eda_outputs/               ← Generated charts
├── src/
│   ├── data/
│   │   ├── loader.py              ← Load and join dataset tables
│   │   └── preprocessor.py       ← Cleaning, encoding, imputation
│   ├── ml/
│   │   ├── train.py               ← Model training pipeline
│   │   ├── predict.py             ← Inference and scoring
│   │   └── evaluate.py            ← Metrics: ROC-AUC, PR-AUC
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py           ← NL → SQL using Groq API
│   │   ├── query_runner.py        ← Execute and return SQL results
│   │   └── prompt_templates.py    ← Versioned prompt templates
│   └── utils/
│       ├── logger.py              ← Logging setup
│       ├── config.py              ← Configuration settings
│       └── helpers.py             ← Utility/helper functions
├── sql/
│   └── schema.sql                 ← Structured SQL DB schema (reference)
├── app.py                         ← Streamlit UI entry point
├── Dockerfile                     ← Docker image definition
├── docker-compose.yml             ← Multi-container orchestration
├── requirements.txt               ← Python dependencies
├── .env.example                   ← Required environment variables
├── .gitignore
└── README.md
```