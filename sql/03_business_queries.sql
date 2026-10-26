-- ===========================================================================
-- 03_business_queries.sql  (retail churn, SQLite)
-- The BI layer: every churn-by-segment number in the notebooks and the
-- dashboard comes from one of these named queries, so the SQL a reviewer
-- reads is exactly the SQL that produced the chart.
--
-- Format: each query starts with a "-- name: <query_name>" line and runs to
-- the next one. src/data_preprocessing.py::run_named_query() loads them.
--
-- Written against 01_create_schema_retail.sql (customers, customer_features)
-- and 02_data_quality_checks.sql.
--
-- Revenue is measured as spend_per_year (lifetime spend / years as customer).
-- It is a spend proxy, not profit: multiply by gross margin for profit.
-- ===========================================================================
 
-- name: kpi_overview
SELECT
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 2)                   AS churn_rate_pct,
    ROUND(SUM(spend_per_year), 2)                       AS annual_revenue,
    ROUND(SUM(CASE WHEN target_churn = 1 THEN spend_per_year ELSE 0 END), 2)
                                                        AS annual_revenue_lost,
    ROUND(SUM(CASE WHEN target_churn = 1 THEN spend_per_year ELSE 0 END)
          / NULLIF(SUM(spend_per_year), 0) * 100, 2)    AS revenue_churn_pct,
    ROUND(AVG(years_as_customer), 1)                    AS avg_tenure_years
FROM customer_features;
 
-- name: churn_by_tenure_bucket
SELECT
    tenure_bucket                                       AS segment,
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct,
    ROUND(AVG(spend_per_year), 2)                       AS avg_spend_per_year
FROM customer_features
GROUP BY tenure_bucket
ORDER BY MIN(years_as_customer);
 
-- name: churn_by_age_bucket
SELECT
    age_bucket                                          AS segment,
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct,
    ROUND(AVG(spend_per_year), 2)                       AS avg_spend_per_year
FROM customer_features
GROUP BY age_bucket
ORDER BY MIN(age);
 
-- name: churn_by_gender
SELECT
    gender                                              AS segment,
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct
FROM customers
GROUP BY gender
ORDER BY churn_rate_pct DESC;
 
-- name: churn_by_satisfaction
SELECT
    satisfaction_score                                  AS segment,
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct
FROM customers
GROUP BY satisfaction_score
ORDER BY satisfaction_score;
 
-- name: churn_by_recency
-- Dormant = no purchase in the last 90 days (see is_dormant in the schema).
SELECT
    CASE WHEN is_dormant = 1 THEN 'Dormant (>90 days)'
         ELSE 'Active (<=90 days)'
    END                                                 AS segment,
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct,
    ROUND(SUM(CASE WHEN target_churn = 1 THEN spend_per_year ELSE 0 END), 0)
                                                        AS annual_revenue_lost
FROM customer_features
GROUP BY is_dormant
ORDER BY churn_rate_pct DESC;
 
-- name: churn_by_support_contacts
-- Does contacting support more often go with higher churn?
SELECT
    num_of_support_contacts                             AS segment,
    COUNT(*)                                            AS customers,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct
FROM customers
GROUP BY num_of_support_contacts
ORDER BY num_of_support_contacts;
 
-- name: churn_by_return_rate
-- Returns as a share of purchases, in three bands.
SELECT
    CASE WHEN return_rate = 0   THEN 'No returns'
         WHEN return_rate <= 0.1 THEN 'Low (up to 10%)'
         ELSE 'High (over 10%)'
    END                                                 AS segment,
    COUNT(*)                                            AS customers,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct
FROM customer_features
GROUP BY segment
ORDER BY churn_rate_pct DESC;
 
-- name: churn_by_promotion_response
SELECT
    promotion_response                                  AS segment,
    COUNT(*)                                            AS customers,
    SUM(target_churn)                                   AS churned,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct,
    ROUND(AVG(spend_per_year), 2)                       AS avg_spend_per_year
FROM customer_features
GROUP BY promotion_response
ORDER BY churn_rate_pct DESC;
 
-- name: churn_by_email_opt_in
-- Matters for retention: opted-out customers cannot be reached by email offers.
SELECT
    email_opt_in                                        AS segment,
    COUNT(*)                                            AS customers,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct
FROM customers
GROUP BY email_opt_in
ORDER BY churn_rate_pct DESC;
 
-- name: churn_dormant_x_unhappy
-- Two behavioural signals crossed: no recent purchase and low satisfaction.
SELECT
    CASE WHEN is_dormant = 1 THEN 'Dormant' ELSE 'Active' END      AS recency,
    CASE WHEN is_unhappy = 1 THEN 'Unhappy (<=2)' ELSE 'Satisfied' END AS satisfaction,
    COUNT(*)                                            AS customers,
    ROUND(AVG(target_churn) * 100, 1)                   AS churn_rate_pct
FROM customer_features
GROUP BY is_dormant, is_unhappy
ORDER BY churn_rate_pct DESC;
 
-- name: churn_by_tenure_year
-- Churn by year of tenure plus a running share of all churners, using window
-- functions: how front-loaded is churn?
WITH by_year AS (
    SELECT years_as_customer,
           COUNT(*)           AS customers,
           SUM(target_churn)  AS churned
    FROM customers
    GROUP BY years_as_customer
)
SELECT
    years_as_customer,
    customers,
    churned,
    ROUND(churned * 100.0 / customers, 1)               AS churn_rate_pct,
    ROUND(SUM(churned) OVER (ORDER BY years_as_customer) * 100.0
          / NULLIF(SUM(churned) OVER (), 0), 1)         AS cumulative_share_of_churners_pct
FROM by_year
ORDER BY years_as_customer;
 
-- name: top_revenue_churners_by_tenure_bucket
-- The three highest-spending customers lost in each tenure bucket.
SELECT tenure_bucket, customer_id, years_as_customer, spend_per_year, rank_in_bucket
FROM (
    SELECT tenure_bucket, customer_id, years_as_customer, spend_per_year,
           ROW_NUMBER() OVER (PARTITION BY tenure_bucket
                              ORDER BY spend_per_year DESC) AS rank_in_bucket
    FROM customer_features
    WHERE target_churn = 1
)
WHERE rank_in_bucket <= 3
ORDER BY tenure_bucket, rank_in_bucket;
 
-- name: retention_targets
-- Needs the customer_scores table written by the scoring step. Active
-- customers only (churned ones are already gone), ranked by expected loss.
SELECT
    s.customer_id,
    s.churn_probability,
    s.risk_level,
    s.expected_loss,
    s.expected_net_gain,
    s.top_reasons,
    f.tenure_bucket,
    f.years_as_customer,
    f.spend_per_year,
    f.satisfaction_score,
    f.last_purchase_days_ago,
    f.email_opt_in
FROM customer_scores s
JOIN customer_features f USING (customer_id)
WHERE f.target_churn = 0
ORDER BY s.expected_loss DESC;
 
-- name: risk_summary
SELECT
    s.risk_level,
    COUNT(*)                                            AS customers,
    ROUND(AVG(s.churn_probability) * 100, 1)            AS avg_churn_probability_pct,
    ROUND(SUM(s.expected_loss), 0)                      AS expected_loss,
    ROUND(SUM(s.churn_probability * f.spend_per_year), 0)
                                                        AS expected_annual_revenue_at_risk
FROM customer_scores s
JOIN customer_features f USING (customer_id)
WHERE f.target_churn = 0
GROUP BY s.risk_level
ORDER BY avg_churn_probability_pct DESC;