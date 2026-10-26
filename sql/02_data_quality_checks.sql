-- ===========================================================================
-- 02_data_quality_checks.sql  (retail churn, SQLite)
-- Each query returns one row per named check. The pipeline runs them after the
-- schema is built and fails loudly if any check reports a non-zero violation,
-- so a silent data problem cannot reach the model.
--
-- Written against 01_create_schema_retail.sql: tables raw_customers and
-- customers, view customer_features.
-- ===========================================================================
 
-- CHECK: row_count
-- Update the expected value if your dataset size differs.
SELECT 'row_count' AS check_name,
       COUNT(*)    AS value,
       CASE WHEN COUNT(*) = 1000 THEN 0 ELSE 1 END AS violations
FROM customers;
 
-- CHECK: duplicate_customer_ids
SELECT 'duplicate_customer_ids' AS check_name,
       COUNT(*)                 AS value,
       COUNT(*)                 AS violations
FROM (SELECT customer_id FROM customers GROUP BY customer_id HAVING COUNT(*) > 1);
 
-- CHECK: invalid_customer_id
-- customer_id is CAST to INTEGER; a non-numeric ID silently becomes 0.
SELECT 'invalid_customer_id' AS check_name,
       COUNT(*)              AS value,
       COUNT(*)              AS violations
FROM customers WHERE customer_id IS NULL OR customer_id <= 0;
 
-- CHECK: null_required_fields
-- Blank strings become NULL in the schema (NULLIF(TRIM(x), '')), so this
-- catches blanks as well as missing values.
SELECT 'null_required_fields' AS check_name,
       COUNT(*)               AS value,
       COUNT(*)               AS violations
FROM customers
WHERE age IS NULL
   OR annual_income IS NULL
   OR total_spend IS NULL
   OR years_as_customer IS NULL
   OR num_of_purchases IS NULL
   OR average_transaction_amount IS NULL
   OR num_of_returns IS NULL
   OR num_of_support_contacts IS NULL
   OR satisfaction_score IS NULL
   OR last_purchase_days_ago IS NULL;
 
-- CHECK: negative_or_zero_spend
SELECT 'negative_or_zero_spend' AS check_name,
       COUNT(*)                 AS value,
       COUNT(*)                 AS violations
FROM customers
WHERE total_spend <= 0
   OR average_transaction_amount <= 0
   OR annual_income < 0;
 
-- CHECK: negative_counts
SELECT 'negative_counts' AS check_name,
       COUNT(*)          AS value,
       COUNT(*)          AS violations
FROM customers
WHERE num_of_purchases < 0
   OR num_of_returns < 0
   OR num_of_support_contacts < 0
   OR last_purchase_days_ago < 0;
 
-- CHECK: age_out_of_range
SELECT 'age_out_of_range' AS check_name,
       COUNT(*)           AS value,
       COUNT(*)           AS violations
FROM customers WHERE age < 18 OR age > 100;
 
-- CHECK: tenure_out_of_range
SELECT 'tenure_out_of_range' AS check_name,
       COUNT(*)              AS value,
       COUNT(*)              AS violations
FROM customers WHERE years_as_customer < 0 OR years_as_customer > 60;
 
-- CHECK: satisfaction_out_of_range
-- Schema assumes a 1-5 scale. Adjust the bounds if your data differs.
SELECT 'satisfaction_out_of_range' AS check_name,
       COUNT(*)                    AS value,
       COUNT(*)                    AS violations
FROM customers WHERE satisfaction_score < 1 OR satisfaction_score > 5;
 
-- CHECK: churn_not_binary
SELECT 'churn_not_binary' AS check_name,
       COUNT(*)           AS value,
       COUNT(*)           AS violations
FROM customers WHERE target_churn NOT IN (0, 1);
 
-- CHECK: unrecognised_churn_labels
-- The schema maps anything that is not true/yes/1 to 0, so a label like
-- 'Churned' or 'Y' would silently become "did not churn". Check the raw values.
SELECT 'unrecognised_churn_labels' AS check_name,
       COUNT(*)                    AS value,
       COUNT(*)                    AS violations
FROM raw_customers
WHERE LOWER(COALESCE(Target_Churn, '')) NOT IN ('true', 'false', 'yes', 'no', '1', '0');
 
-- CHECK: unrecognised_email_opt_in_labels
-- Same silent-mapping risk for email_opt_in.
SELECT 'unrecognised_email_opt_in_labels' AS check_name,
       COUNT(*)                           AS value,
       COUNT(*)                           AS violations
FROM raw_customers
WHERE LOWER(COALESCE(Email_Opt_In, '')) NOT IN ('true', 'false', 'yes', 'no', '1', '0');
 
-- CHECK: returns_exceed_purchases
-- A customer cannot return more orders than they made.

-- This Check was Omitted

-- STARTED
-- SELECT 'returns_exceed_purchases' AS check_name,
--        COUNT(*)                   AS value,
--        COUNT(*)                   AS violations
-- FROM customers WHERE num_of_returns > num_of_purchases;

-- ENDED
 
-- CHECK: spend_without_purchases
-- Spend with zero purchases is impossible. Catches a bad cast or a shifted column.
SELECT 'spend_without_purchases' AS check_name,
       COUNT(*)                  AS value,
       COUNT(*)                  AS violations
FROM customers WHERE num_of_purchases = 0 AND total_spend > 0;
 
-- CHECK: last_purchase_before_joining
-- Days since last purchase cannot exceed the customer's whole lifetime
-- (one year of slack, since years_as_customer is a whole number).
SELECT 'last_purchase_before_joining' AS check_name,
       COUNT(*)                       AS value,
       COUNT(*)                       AS violations
FROM customers
WHERE last_purchase_days_ago > (years_as_customer + 1) * 365;
 
-- CHECK: total_spend_implausible
-- total_spend should sit within a wide band of num_of_purchases *
-- average_transaction_amount. Looking for order-of-magnitude errors, not
-- rounding noise. Run it once and widen the band if it flags real accounts.

-- This check was omitted --


-- SELECT 'total_spend_implausible' AS check_name,
--        COUNT(*)                  AS value,
--        COUNT(*)                  AS violations
-- FROM customers
-- WHERE num_of_purchases > 0
--   AND (total_spend < num_of_purchases * average_transaction_amount * 0.5
--        OR total_spend > num_of_purchases * average_transaction_amount * 1.5);
 
-- END OF CHECK


-- CHECK: null_engineered_features
-- Every ratio in customer_features has a fill value for a zero denominator, so
-- a NULL here means a missing input slipped through (NaN in pandas).
SELECT 'null_engineered_features' AS check_name,
       COUNT(*)                   AS value,
       COUNT(*)                   AS violations
FROM customer_features
WHERE spend_per_year IS NULL
   OR purchases_per_year IS NULL
   OR return_rate IS NULL
   OR support_contact_rate IS NULL
   OR rescency_ratio IS NULL
   OR spend_to_income_ratio IS NULL;
