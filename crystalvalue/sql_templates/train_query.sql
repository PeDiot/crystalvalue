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
      (SELECT MIN(DATE({date_column})) FROM {project_id}.{dataset_id}.{table_name}),
      INTERVAL {days_lookback} DAY
    )
  ),
  Target AS (
    SELECT
      CustomerWindows.*,
      SUM(IFNULL(TX_DATA.{value_column}, 0)) AS future_value,
    FROM
      CustomerWindows
    LEFT JOIN
      {project_id}.{dataset_id}.{table_name} AS TX_DATA
      ON (
        CAST(TX_DATA.{customer_id_column} AS STRING) = CustomerWindows.customer_id
        AND DATE(TX_DATA.{date_column})
          BETWEEN CustomerWindows.lookahead_start
          AND CustomerWindows.lookahead_stop)
    GROUP BY
      1, 2, 3, 4, 5
  )
  , Dataset AS (
    SELECT
      Target.*,
      CASE WHEN future_value > 1 THEN 1 ELSE 0 END AS future_value_classification,
      CASE
        WHEN EXTRACT(YEAR FROM Target.window_date) = 2024 
            AND EXTRACT(MONTH FROM Target.window_date) = 3
          THEN 'TEST'
        WHEN EXTRACT(YEAR FROM Target.window_date) != 2024 
            OR EXTRACT(MONTH FROM Target.window_date) != 3
          THEN
            CASE
              WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(Target.customer_id))), 100))
                  BETWEEN 0 AND 15
                THEN 'VALIDATE'
              ELSE 'TRAIN'
            END
        END AS predefined_split_column,
      IFNULL(
        DATE_DIFF(Target.window_date, MAX(DATE(TX_DATA.{date_column})), DAY),
        {days_lookback}) AS days_since_last_transaction,
      IFNULL(
        DATE_DIFF(Target.window_date, MIN(DATE(TX_DATA.{date_column})), DAY),
        {days_lookback}) AS days_since_first_transaction,
      COUNT(*) AS count_transactions,
      {features_sql}
    FROM
      Target
    JOIN
      {project_id}.{dataset_id}.{table_name} AS TX_DATA
      ON (
        CAST(TX_DATA.{customer_id_column} AS STRING) = Target.customer_id
        AND DATE(TX_DATA.{date_column}) BETWEEN Target.lookback_start AND DATE(Target.window_date))
    GROUP BY
      1, 2, 3, 4, 5, 6, 7
  )
SELECT * FROM Dataset;