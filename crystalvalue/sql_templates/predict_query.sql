WITH
  FirstPurchaseDates AS (
    SELECT 
      CAST({customer_id_column} AS STRING) AS customer_id,
      MIN(DATE({date_column})) AS first_purchase_date
    FROM {project_id}.{dataset_id}.{table_name}
    GROUP BY 1
  ),
  CustomerWindows AS (
    SELECT DISTINCT
      FirstPurchaseDates.customer_id,
      FirstPurchaseDates.first_purchase_date AS window_date,
      DATE_SUB(FirstPurchaseDates.first_purchase_date, INTERVAL {days_lookback} day) AS lookback_start,
      DATE_ADD(FirstPurchaseDates.first_purchase_date, INTERVAL 1 day) AS lookahead_start,
      DATE_ADD(FirstPurchaseDates.first_purchase_date, INTERVAL {days_lookahead} day) AS lookahead_stop
    FROM FirstPurchaseDates
    WHERE FirstPurchaseDates.first_purchase_date >= DATE_ADD(
      (SELECT MIN({date_column}) FROM {project_id}.{dataset_id}.{table_name}),
      INTERVAL {days_lookback} DAY
    )
  )
  , Dataset AS (
    SELECT
      CustomerWindows.*,
      IFNULL(
        DATE_DIFF(CustomerWindows.window_date, MAX(DATE(TX_DATA.{date_column})), DAY),
        {days_lookback}) AS days_since_last_transaction,
      IFNULL(
        DATE_DIFF(CustomerWindows.window_date, MIN(DATE(TX_DATA.{date_column})), DAY),
        {days_lookback}) AS days_since_first_transaction,
      COUNT(*) AS count_transactions,
      {features_sql}
    FROM
      CustomerWindows
    JOIN
      {project_id}.{dataset_id}.{table_name} AS TX_DATA
      ON (
        CAST(TX_DATA.{customer_id_column} AS STRING) = CustomerWindows.customer_id
        AND DATE(TX_DATA.{date_column}) BETWEEN CustomerWindows.lookback_start AND DATE(CustomerWindows.window_date))
    GROUP BY
      1, 2, 3, 4, 5
  )
SELECT 
  customer_id,
  window_date,
  lookback_start,
  lookahead_start,
  lookahead_stop,
  future_value,
  future_value_classification,
  predefined_split_column,
  avg_value AS value, 
  avg_value_cat1 AS value_cat1,
  avg_value_cat2 AS value_cat2,
  avg_value_cat3 AS value_cat3,
  avg_value_cat4 AS value_cat4,
  avg_value_cat5 AS value_cat5,
  avg_value_cat6 AS value_cat6,
  avg_value_uncategorized AS value_uncategorized,
  unique_list_shipping_address_zip AS shipping_address_zip,
  unique_list_order_index AS order_index
FROM Dataset;