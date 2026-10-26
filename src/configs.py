"""
Central Configuration center for the Customer Churn Prediction & BI System

Every thing a business user might want to change - file locations,
economics of retention campaign, model hyperparameters - to avoid 
hard coding a magic number
"""

from __future__ import annotations
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

#--------------------------------------------
# Paths
#--------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT/ "models"
REPORTS_DIR = PROJECT_ROOT /"reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

RAW_CSV = RAW_DIR / "online_retail_customer_churn.csv"
DB_PATH = DATA_DIR / "churn.db"
FEATURES_CSV = PROCESSED_DIR / "customers_features.csv"


MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METRICS_JSON = REPORTS_DIR / "model_comparison.json"
THRESHOLD_JSON = REPORTS_DIR / "threshold_economics.json"
SCORED_CSV = PROCESSED_DIR / "customers_scored.csv"
GLOBAL_IMPORTANCE_CSV = REPORTS_DIR / "global_importance.csv"
SEGMENTS_CSV = PROCESSED_DIR / "customer_segments.csv"
LLM_EXAMPLES_JSON = REPORTS_DIR / "llm_examples.json"


for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SQL_DIR):
    _d.mkdir(parents=True, exist_ok= True)

#----------------------------------------------
# Modelling
#----------------------------------------------

TARGET = "target_churn"
ID_COL = "customer_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 25

SELECTION_METRIC = "roc_auc"

BASE_NUMERIC = [
    'age', 'annual_income', 
    'total_spend', 'years_as_customer', 
    'num_of_purchases', 'average_transaction_amount', 
    'num_of_returns', 'num_of_support_contacts', 
    'satisfaction_score', 'last_purchase_days_ago'
]

BASE_CATEGORICAL = ["gender", 'promotion_response', 'email_opt_in']


ENGINEERED_NUMERIC = [
    "return_rate",              # num_of_returns / num_of_purchases
    "purchases_per_year",       # num_of_purchases / years_as_customer
    "spend_per_year",           # total_spend / years_as_customer
    "support_contact_rate",     # num_of_support_contacts / num_of_purchases
    "rescency_ratio",            # last_purchase_days_ago / (365 / purchases_per_year)
    "spend_to_income_ratio"     # total_spend / annual_income
]

NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = BASE_CATEGORICAL
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Scored columns published to the SQL layer as the `customer_scores` table.
SCORE_TABLE_COLUMNS = [
    "customer_id", "churn_probability", "risk_level", "customer_value", "expected_loss",
    "expected_net_gain", "contact_recommended", "top_reasons", "recommended_action", "segment",
]

# Columns a raw retail customer upload must contain.
RAW_REQUIRED_COLUMNS = [
    'Customer_ID','Age','Gender',
    'Annual_Income','Total_Spend','Years_as_Customer',
    'Num_of_Purchases','Average_Transaction_Amount','Num_of_Returns',
    'Num_of_Support_Contacts','Satisfaction_Score','Last_Purchase_Days_Ago',
    'Email_Opt_In','Promotion_Response','Target_Churn'
]


#----------------------------------------------
# Campaign economics
#----------------------------------------------
@dataclass(frozen=True)
class Economics:
    """
    The money model behind the retention campaign.

    The model outputs a probablity, turning that probability into a *decision*
    ("do we call this customer ?") requires a price list. These are
    the assumptions the optimal threshold is derived from - all of them are
    are exposed as sliders in the dashboard so a manager can test their own numbers.
    """

    # One-off cost of extending a retention offer (discount, credit, call time)
    # Paid for every customer we contact, whether or not they were going to leave.
    offer_cost: float = 20.0
    
    save_rate: float = 0.30
    
    horizon_months: int = 12
    
    gross_margin: float = 0.30

    def customer_value(self, spend_per_year):
       """Margin value of keeping a customer over the planning horizon."""
       return np.asarray(spend_per_year, dtype=float) * (self.horizon_months / 12) * self.gross_margin

    def as_dict(self) -> dict:
        return asdict(self)
ECONOMICS = Economics()

#----------------------------------------------
# Risk Banding (applied to churn probablity). Bands are for communication:
# Who gets contacted is decided by money-optimal threshold
#----------------------------------------------

RISK_BANDS = [(0.60, 'HIGH'), (0.30, 'MEDIUM'), (0.00, 'LOW')]

def risk_level(prob: float) -> str:
    for cutoff, label in RISK_BANDS:
        if prob >= cutoff:
            return label
    return "LOW"

# --------------------------------------------------------------------------- #
# LLM (retention summaries). Optional: without an API key the app falls back to
# a deterministic template, so nothing breaks on a fresh clone.
# --------------------------------------------------------------------------- #
LLM_MODEL = "claude-haiku-5-5"

# --------------------------------------------------------------------------- #
# Human-readable names for model features (used in charts and explanations)
# --------------------------------------------------------------------------- #
FEATURE_LABELS = {
    'customer_id': 'Customer ID',
    'age': 'Age',
    'gender': 'Gender',
    'annual_income': 'Annual Income',
    'total_spend': 'Total Spend',
    'spend_to_income_ratio':'Spend To Income',
    'purchases_per_year': 'Purchases Per Year',
    'spend_per_year': 'Spend Per Year',
    'years_as_customer': 'Years as Customer',
    'num_of_purchases': 'Num of Purchases',
    'average_transaction_amount':'Average Transaction Amount',
    'num_of_returns': 'Num of Returns',
    'return_rate': 'Return Rate',
    'num_of_support_contacts': 'Num of Support Contacts',
    'support_contact_rate': 'Support Contact Rate',
    'satisfaction_score': 'Satisfaction Score',
    'last_purchase_days_ago': 'Last_Purchase Days Ago',
    'rescency_ratio':'Rescency Ratio',
    'email_opt_in': 'Email Opt In',
    'promotion_response': 'Promotion Response',
    'target_churn': 'Target Churn'
}