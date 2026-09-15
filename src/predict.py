from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import mlflow


CATALOG = "main"
SCHEMA = "ml_demo"

FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features"
PREDICTION_TABLE = f"{CATALOG}.{SCHEMA}.predictions"

MODEL_NAME = f"{CATALOG}.{SCHEMA}.customer_purchase_model"


FEATURE_COLUMNS = [
    "age",
    "income",
    "previous_purchases",
    "avg_purchase",
    "total_spending",
]


def main():
    spark = SparkSession.builder.getOrCreate()

    features = spark.table(FEATURE_TABLE)

    if spark.catalog.tableExists(PREDICTION_TABLE):
        predicted_ids = spark.table(PREDICTION_TABLE).select(
            "customer_id"
        )

        new_data = features.join(
            predicted_ids,
            on="customer_id",
            how="left_anti",
        )
    else:
        new_data = features

    new_count = new_data.count()

    print(f"New records to predict: {new_count}")

    if new_count == 0:
        print("No new records.")
        return

    mlflow.set_registry_uri("databricks-uc")

    model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/latest"
    )

    pandas_data = new_data.select(
        ["customer_id"] + FEATURE_COLUMNS
    ).toPandas()

    X = pandas_data[FEATURE_COLUMNS]

    predictions = model.predict(X)

    pandas_data["prediction"] = predictions

    result = spark.createDataFrame(pandas_data)

    result = (
        result
        .withColumn(
            "prediction_timestamp",
            F.current_timestamp(),
        )
        .withColumn(
            "model_name",
            F.lit(MODEL_NAME),
        )
    )

    result = result.select(
        "customer_id",
        "prediction",
        "prediction_timestamp",
        "model_name",
    )

    (
        result
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(PREDICTION_TABLE)
    )

    print(
        f"Added {new_count} predictions."
    )


if __name__ == "__main__":
    main()