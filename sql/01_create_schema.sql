-- Retail customer churn schema matching src/configs.py.
-- Requires raw_customers to be loaded from configs.RAW_CSV first.

DROP VIEW IF EXISTS customer_features;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers AS
SELECT
    NULLIF(TRIM(Customer_ID), '') AS customer_id,
    CAST(NULLIF(TRIM(Age), '') AS INTEGER) AS age,
    NULLIF(TRIM(Gender), '') AS gender,
    CAST(NULLIF(TRIM(Annual_Income), '') AS REAL) AS annual_income,
    CAST(NULLIF(TRIM(Total_Spend), '') AS REAL) AS total_spend,
    CAST(NULLIF(TRIM(Years_as_Customer), '') AS REAL) AS years_as_customer,
    CAST(NULLIF(TRIM(Num_of_Purchases), '') AS REAL) AS num_of_purchases,
    CAST(NULLIF(TRIM(Average_Transaction_Amount), '') AS REAL) AS average_transaction_amount,
    CAST(NULLIF(TRIM(Num_of_Returns), '') AS REAL) AS num_of_returns,
    CAST(NULLIF(TRIM(Num_of_Support_Contacts), '') AS REAL) AS num_of_support_contacts,
    CAST(NULLIF(TRIM(Satisfaction_Score), '') AS REAL) AS satisfaction_score,
    CAST(NULLIF(TRIM(Last_Purchase_Days_Ago), '') AS REAL) AS last_purchase_days_ago,
    CASE
        WHEN LOWER(TRIM(Email_Opt_In)) IN ('1','true','yes','y') THEN 'Yes'
        WHEN LOWER(TRIM(Email_Opt_In)) IN ('0','false','no','n') THEN 'No'
        ELSE NULLIF(TRIM(Email_Opt_In), '')
    END AS email_opt_in,
    NULLIF(TRIM(Promotion_Response), '') AS promotion_response,
    CASE
        WHEN LOWER(TRIM(Target_Churn)) IN ('1','true','yes','y','churned') THEN 1
        WHEN LOWER(TRIM(Target_Churn)) IN ('0','false','no','n','retained') THEN 0
        ELSE CAST(NULLIF(TRIM(Target_Churn), '') AS INTEGER)
    END AS target_churn
FROM raw_customers
WHERE NULLIF(TRIM(Customer_ID), '') IS NOT NULL
GROUP BY TRIM(Customer_ID);

CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_target_churn ON customers(target_churn);

CREATE VIEW customer_features AS
WITH base AS (
    SELECT c.*,
        CASE WHEN COALESCE(num_of_purchases, 0) > 0
             THEN COALESCE(num_of_returns, 0) / num_of_purchases ELSE 0.0 END AS return_rate,
        CASE WHEN COALESCE(years_as_customer, 0) > 0
             THEN COALESCE(num_of_purchases, 0) / years_as_customer ELSE 0.0 END AS purchases_per_year,
        CASE WHEN COALESCE(years_as_customer, 0) > 0
             THEN COALESCE(total_spend, 0) / years_as_customer
             ELSE COALESCE(total_spend, 0) END AS spend_per_year,
        CASE WHEN COALESCE(num_of_purchases, 0) > 0
             THEN COALESCE(num_of_support_contacts, 0) / num_of_purchases ELSE 0.0 END AS support_contact_rate,
        CASE WHEN COALESCE(annual_income, 0) > 0
             THEN COALESCE(total_spend, 0) / annual_income ELSE 0.0 END AS spend_to_income_ratio
    FROM customers AS c
)
SELECT base.*,
    CASE WHEN COALESCE(purchases_per_year, 0) > 0
         THEN COALESCE(last_purchase_days_ago, 0) / (365.0 / purchases_per_year)
         ELSE 0.0 END AS rescency_ratio
FROM base;