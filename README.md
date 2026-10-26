Retail Customer Churn Intelligence
An end-to-end machine learning and business intelligence system that predicts customer churn, explains why customers are at risk, and helps retention teams decide who to contact, what action to take, and whether the campaign is financially worthwhile.

The project combines a churn-scoring pipeline, customer-level explanations, SQL analytics, model evaluation, and an interactive Streamlit dashboard.
Project status: Portfolio project. Results shown in the dashboard are based on the current dataset and pipeline configuration; they should not be interpreted as proof of production performance or causal campaign impact.

Table of Contents
- Overview
- Dashboard Features
- How It Works
- Model Evaluation
- Campaign Economics
- Technology Stack
- Project Architecture
- Getting Started
- Data Requirements
- Key Business Concepts
- Limitations and Future Improvements
Overview
Customer churn can reduce recurring revenue and increase the cost of acquiring replacement customers. A churn probability alone, however, does not tell a business which customer is worth contacting or whether a retention offer will generate positive value.

Retail Customer Churn Intelligence connects predictive analytics with retention decision-making:
1. Estimate each customer's probability of churn.
2. Rank customers using churn risk and expected value at risk.
3. Explain important churn signals using SHAP.
4. Compare retention targeting strategies at different contact thresholds.
5. Estimate campaign costs, retained value, and net value under configurable assumptions.
6. Explore results through interactive dashboards and a read-only SQL Lab.
Project goals
- Turn customer records into actionable retention priorities.
- Make model predictions more interpretable at both customer and portfolio level.
- Compare classification metrics rather than relying on accuracy alone.
- Connect model scores to explicit business assumptions and campaign economics.
- Provide an interface for analysts and business stakeholders to explore the results.
Dashboard Features
1. Overview — Customer Retention Command Center
A portfolio-level summary of customers, predicted churn, recommended contacts, and estimated financial exposure.
- Customer and churn KPIs
- Contact-threshold summary
- Expected margin-loss and campaign-value estimates
- Summary of the active model and scoring approach
2. Retention Targets
A ranked list of customers to investigate for possible retention outreach.
- Churn probability and risk level
- Expected loss and estimated net gain if contacted
- Customer-level SHAP reasons
- Recommended next-best action
- AI-written customer brief, when configured
- Filtering by risk, segment, and promotion response
- CSV export for downstream analysis
The dashboard ranks customers using expected loss, combining estimated churn probability with customer value. A high probability of churn does not automatically mean a customer is the most valuable one to contact.
3. Customer-Level Explanation
An individual customer view that brings the prediction and its business context together.
- Churn probability gauge
- Customer profile and behavioural indicators
- Main factors associated with the prediction
- SHAP impact chart
- Suggested retention actions and offer comparisons
SHAP explanations describe how model features contribute to a prediction. They do not establish that a feature causes churn or that changing it will necessarily prevent churn.
4. Campaign Economics
A decision-support page for testing retention assumptions.

Configurable assumptions include:
- Offer cost per contact
- Save rate among contacted customers who would otherwise churn
- Customer-value horizon
- Gross margin
The page compares targeting rules and plots estimated retained value, campaign cost, and net value across contact thresholds. It also reports metrics such as customers contacted, churners reached, precision, and recall.

These are scenario estimates based on the selected assumptions, not guaranteed campaign outcomes.
5. Churn Drivers & Segments
Portfolio-level analysis of model drivers and customer groups.
- Mean absolute SHAP values
- Permutation feature importance
- Feature-dependence plots
- Behavioural segments created using clustering
- Segment-level churn-rate and customer-behaviour comparisons
SHAP and permutation importance answer different questions. SHAP summarizes feature contributions to model predictions; permutation importance measures how model performance changes when a feature is shuffled.
6. Model Performance
An evaluation workspace for comparing candidate classifiers.
- Cross-validation and held-out test ROC-AUC
- Precision, recall, F1, accuracy, and PR-AUC where available
- ROC and precision-recall curves
- Confusion matrix at the selected threshold
- Probability calibration plot
- Comparisons of class weighting and engineered features
The decision threshold is a business choice as well as a modelling choice. A threshold that increases recall may also increase false-positive contacts and campaign cost.
7. AI Copilot & SQL Lab
An interface for exploring customer and churn data.
- SQL query templates
- Read-only query execution
- Schema exploration
- Tabular results and visual summaries
- Optional natural-language summaries or AI-generated briefs when the required model/API configuration is available
Security note: Keep database credentials and API keys out of source control. Read-only query controls should not be treated as a substitute for database permissions and query validation.
8. Score New Customers
Upload a CSV of retail customer records to score using the existing preprocessing and prediction pipeline.
- CSV upload
- Churn probability and risk classification
- Contact-threshold indicator
- Retention-related estimates where supported by the configured pipeline
Input columns must match the feature schema expected by the model. The target column is optional for scoring new, unlabeled customers.

Modelling and evaluation approach
The dashboard indicates that the project compares multiple model families, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
The champion model shown in the current dashboard is Decision Tree, with a displayed held-out test ROC-AUC of approximately 0.521. This is only modestly above random-ranking performance, so the current model should be treated as an experimental baseline rather than a reliable production churn model.

The dashboard also indicates the use of stratified cross-validation, out-of-fold scores, threshold comparison, and SHAP-based explanations. Confirm the implementation and data-splitting details in the code before describing them as guarantees.
Model Evaluation
The project evaluates models using several metrics because no single metric fully describes churn-model usefulness.
Metric	What it tells us
ROC-AUC	How well the model ranks churners above non-churners across thresholds
PR-AUC	Precision-recall performance, particularly useful when churn is uncommon
Precision	Of customers predicted to churn, the proportion who actually churned
Recall	Of actual churners, the proportion identified by the model
F1-score	Harmonic mean of precision and recall
Confusion matrix	Counts of true positives, true negatives, false positives, and false negatives
Brier score	Overall squared error of predicted probabilities; lower is better
Calibration plot	Whether predicted probabilities align with observed churn frequencies


Current result to interpret carefully
The screenshots show a Decision Tree test ROC-AUC of about 0.521. This suggests limited discrimination on the displayed test set. A model can produce detailed explanations and a polished dashboard while still having weak predictive performance; the explanations do not make the underlying predictions more accurate.

Before using scores to make real retention decisions, investigate feature quality, label definition, leakage, train/test representativeness, calibration, and performance against a simple baseline.
Campaign Economics
The project estimates the potential value of retention outreach under configurable assumptions.

A simplified decision framework is:
- Expected loss = predicted churn probability × estimated customer value
- Campaign cost = number of customers contacted × cost per contact
- Expected retained value = estimated value retained under the chosen save-rate assumption
- Net campaign value = expected retained value − campaign cost
The exact implementation may use additional rules or adjustments; consult the campaign-economics code for the precise formula.

The dashboard compares rules such as contacting nobody, contacting everyone, using a fixed probability threshold, using a money-oriented threshold, and selecting customers by expected value.

Important: These calculations are assumption-driven estimates. The assumed save rate is not evidence that an offer causes customers to stay. A real deployment should validate the assumptions with a controlled retention experiment and calculate incremental value against a control group.
Technology Stack
The project interface and screenshots indicate the following tools and methods:
Area	Technology
Language	Python
Data preparation	pandas, NumPy
Machine learning	scikit-learn, XGBoost
Explainability	SHAP
Visualisation	Plotly
Dashboard	Streamlit
Data querying	SQL, SQLite (as shown in the SQL Lab interface)
Data export	CSV
Optional AI features	Configured language-model/API integration


Check the dependency files and imports in the repository for the exact versions and any additional services.
Project Architecture
The application is organised around separate dashboard views and reusable analytics modules. The screenshots show pages for overview, retention targets, scenario analysis, campaign economics, churn analysis, model performance, SQL Lab, and scoring new customers.

Getting Started
Prerequisites
- Python compatible with the versions used by the project dependencies
- pip
- The project dataset and any required trained model artefacts
- Optional API credentials if AI-generated briefs or Copilot features are enabled

1. Clone the repository
git clone https://github.com/abhi8934/customer_churn_ref.git
cd customer_churn_ref

2. Create and activate a virtual environment
Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
If the repository includes requirements.txt:
python -m pip install --upgrade pip
pip install -r requirements.txt
If dependency installation fails, check the Python version and the project's pinned package versions. Do not install arbitrary latest versions without testing compatibility.

4. Configure paths and environment variables
Review configs.py and the project's environment-variable configuration. Place datasets and model artefacts in the paths expected by the application.

If AI features require an API key, store it in a local .env file or environment variable and ensure that file is excluded from Git. Never commit secrets.

5. Prepare the data and model artefacts
Run the project's documented training or scoring pipeline if model artefacts and reports are not already supplied. Use the actual script names and commands defined in the repository; they may differ from the examples in other projects.

6. Launch the dashboard
If the Streamlit entry point is dashboard/app.py, run:
streamlit run dashboard/app.py
If the repository uses a different entry point, substitute the correct path. Streamlit will print the local URL in the terminal.
Data Requirements
The scoring interface expects a CSV containing the feature columns defined in the project configuration and model preprocessing pipeline.

The screenshots indicate that the analysis uses customer behaviour and value fields such as:
- Customer tenure
- Annual spend or income
- Average transaction amount
- Purchase frequency and number of purchases
- Days since last purchase
- Satisfaction score
- Return rate
- Support-contact behaviour
- Spend-to-income or related value ratios
- Customer segment or promotion-response fields, where supported
The precise required columns, accepted data types, missing-value handling, and categorical levels must be taken from the repository's configured feature schema. Do not assume that every listed field is mandatory for every page.

For reliable scoring, training and inference must use the same feature definitions and preprocessing steps. Avoid including identifiers or post-outcome information as predictive features unless there is a clear, defensible reason.
Key Business Concepts
- Churn probability: The model's estimated likelihood of the defined churn outcome, subject to calibration quality.
- Expected loss: A model-based estimate of customer value exposed to churn.
- Contact threshold: The rule that determines which customers qualify for outreach.
- Precision and recall: The trade-off between concentrating on likely churners and reaching a large share of actual churners.
- SHAP: A method for explaining how features contribute to individual predictions.
- Permutation importance: A measure of how model performance changes when a feature's values are shuffled.
- Out-of-fold scoring: Predictions for training examples generated by models that did not train on those particular examples, when implemented correctly.
- Campaign net value: Estimated retained value minus outreach and offer costs under explicit assumptions.
Limitations and Future Improvements
Current limitations
- The displayed Decision Tree ROC-AUC is approximately 0.521, indicating weak ranking performance on the shown test set.
- Business-value outputs depend on assumptions about margin, customer value, offer cost, and save rate.
- Feature importance and SHAP values describe model behaviour, not causal drivers of churn.
- Segment-level findings depend on the selected features, scaling, and clustering method.
- AI-generated summaries, if enabled, may be incomplete or incorrect and should be reviewed.
- The screenshots do not independently verify deployment security, monitoring, or performance on future data.
Planned improvements
- Improve data validation, label quality, and feature engineering.
- Establish simple baselines and investigate why current ROC-AUC is low.
- Compare calibrated models and select thresholds using business costs.
- Use time-based or otherwise deployment-representative validation where appropriate.
- Test retention interventions with control groups and estimate incremental lift.
- Add automated tests for preprocessing, scoring, SQL queries, and campaign calculations.
- Add reproducible experiment tracking, model versioning, and data/model drift monitoring.
- Improve accessibility, dashboard responsiveness, and documentation.
- Document dataset provenance, licensing, privacy safeguards, and reproducible setup instructions.
Responsible Use
This application is a decision-support tool, not an automated instruction to contact or deny service to a customer. Review recommendations before taking action, minimise personal data, protect customer information, and assess whether features or targeting rules create unfair outcomes. Use the dashboard's estimates alongside business context and validated experiments.
Author
Abhimanyu Pandey
Student project — Retail Analytics, Machine Learning, and Business Intelligence
