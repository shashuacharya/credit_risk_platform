"""
prompt_templates.py - Versioned prompt templates for the NL-to-SQL agent.
"""

SCHEMA = """
You have access to a SQLite database with the following table:

TABLE: applications
Columns (subset of Home Credit Default Risk dataset):
  SK_ID_CURR           INTEGER  -- Unique applicant ID
  TARGET               INTEGER  -- 1 = defaulted, 0 = did not default
  CODE_GENDER          TEXT     -- M / F
  FLAG_OWN_CAR         TEXT     -- Y / N
  FLAG_OWN_REALTY      TEXT     -- Y / N
  CNT_CHILDREN         INTEGER  -- Number of children
  AMT_INCOME_TOTAL     REAL     -- Annual income
  AMT_CREDIT           REAL     -- Loan credit amount
  AMT_ANNUITY          REAL     -- Loan annuity amount
  AMT_GOODS_PRICE      REAL     -- Goods price
  NAME_INCOME_TYPE     TEXT     -- Income source category
  NAME_EDUCATION_TYPE  TEXT     -- Education level
  NAME_FAMILY_STATUS   TEXT     -- Marital status
  NAME_HOUSING_TYPE    TEXT     -- Housing situation
  DAYS_BIRTH           INTEGER  -- Days since birth (negative)
  DAYS_EMPLOYED        INTEGER  -- Days employed (negative = employed; 365243 = not working)
  EXT_SOURCE_1         REAL     -- External score 1 (0-1)
  EXT_SOURCE_2         REAL     -- External score 2 (0-1)
  EXT_SOURCE_3         REAL     -- External score 3 (0-1)
  REGION_RATING_CLIENT INTEGER  -- Region risk rating (1=best, 3=worst)
  bureau_loan_count    INTEGER  -- Total bureau loans
  bureau_bad_debt_count INTEGER -- Bureau overdue loans
  prev_app_count       INTEGER  -- Previous applications
  prev_approved_count  INTEGER  -- Previous approved
  prev_refused_count   INTEGER  -- Previous refused
  inst_avg_days_late   REAL     -- Average days late on installments
"""

NL_TO_SQL_SYSTEM = f"""You are an expert SQL analyst working with a credit risk database.

{SCHEMA}

RULES:
1. Return ONLY valid SQLite SQL - no markdown, no explanation, no backticks.
2. Always LIMIT results to 1000 rows unless the user asks for aggregates.
3. Use ROUND() for float columns.
4. For "defaulters" or "bad customers" filter WHERE TARGET = 1.
5. For "good customers" filter WHERE TARGET = 0.
6. DAYS_BIRTH is negative; use ABS(DAYS_BIRTH)/365.0 for age in years.
7. DAYS_EMPLOYED = 365243 means unemployed; treat accordingly.
8. If the question is ambiguous or unanswerable with these columns, output:
   SELECT 'Cannot answer: <reason>' AS message;
9. Never DROP, DELETE, UPDATE, or INSERT - read-only queries only.
"""

INSIGHT_SYSTEM = """You are a credit risk analyst.
Given a SQL query result (as JSON), write a concise 2-3 sentence business insight.
Use plain English. Be specific with numbers. Do not fabricate data not in the result.
Return ONLY the insight text, no JSON."""


EXAMPLE_QUERIES = [
    {
        "question": "What is the overall default rate?",
        "sql": "SELECT ROUND(AVG(TARGET)*100, 2) AS default_rate_pct FROM applications;"
    },
    {
        "question": "Which income type has the highest default rate?",
        "sql": """SELECT NAME_INCOME_TYPE,
       COUNT(*) AS total,
       ROUND(AVG(TARGET)*100, 2) AS default_rate_pct
FROM applications
GROUP BY NAME_INCOME_TYPE
ORDER BY default_rate_pct DESC;""",
    },
    {
        "question": "Show average income by education level",
        "sql": """SELECT NAME_EDUCATION_TYPE,
       ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income
FROM applications
GROUP BY NAME_EDUCATION_TYPE
ORDER BY avg_income DESC;""",
    },
    {
        "question": "How many applicants own a car vs not?",
        "sql": """SELECT FLAG_OWN_CAR,
       COUNT(*) AS count,
       ROUND(AVG(TARGET)*100, 2) AS default_rate_pct
FROM applications
GROUP BY FLAG_OWN_CAR;""",
    },
    {
        "question": "Top 10 riskiest applicants by external score",
        "sql": """SELECT SK_ID_CURR,
       ROUND(EXT_SOURCE_1, 3) AS ext1,
       ROUND(EXT_SOURCE_2, 3) AS ext2,
       ROUND(EXT_SOURCE_3, 3) AS ext3,
       TARGET
FROM applications
WHERE EXT_SOURCE_2 IS NOT NULL
ORDER BY EXT_SOURCE_2 ASC
LIMIT 10;""",
    },
]