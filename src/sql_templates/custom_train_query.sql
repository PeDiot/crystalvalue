-- Build training dataset for a machine learning model to predict LTV based on first purchase only.
-- The query creates a test set for March 2024 predictions.

-- @param customer_id_column STRING The column containing the customer ID.
-- @param date_column STRING The column containing the transaction date.
-- @param value_column STRING The column containing the value column.
-- @param features_sql STRING The SQL for the features and transformations.

WITH
  FirstPurchases AS (
    SELECT
      CAST({customer_id_column} AS STRING) AS customer_id,
      MIN(DATE({date_column})) AS first_purchase_date,
      {value_column} AS first_purchase_value
    FROM {project_id}.{dataset_id}.{table_name}
    GROUP BY 1, 3
  ),
  March2024Target AS (
    SELECT
      customer_id,
      first_purchase_date,
      first_purchase_value,
      DATE('2024-03-01') AS prediction_date,
      DATE_DIFF(DATE('2024-03-01'), first_purchase_date, DAY) AS days_since_first_purchase,
      -- Calculate future value (LTV) from first purchase until March 2024
      SUM(IFNULL({value_column}, 0)) AS future_value
    FROM FirstPurchases
    LEFT JOIN {project_id}.{dataset_id}.{table_name} AS TX_DATA
      ON CAST(TX_DATA.{customer_id_column} AS STRING) = FirstPurchases.customer_id
      AND DATE(TX_DATA.{date_column}) > FirstPurchases.first_purchase_date
      AND DATE(TX_DATA.{date_column}) <= DATE('2024-03-31')
    GROUP BY 1, 2, 3, 4, 5
  )
SELECT
  customer_id,
  first_purchase_date,
  first_purchase_value,
  prediction_date,
  days_since_first_purchase,
  future_value,
  CASE WHEN future_value > 1 THEN 1 ELSE 0 END AS future_value_classification,
  'TEST' AS predefined_split_column,  -- All data is test data for March 2024
  {features_sql}
FROM March2024Target; 