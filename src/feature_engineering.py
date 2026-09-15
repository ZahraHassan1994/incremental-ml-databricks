from pyspark.sql import SparkSession
from pyspark.sql import functions as F


CATALOG = "main"
SCHEMA = "ml_demo"

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.customer_data"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features"


def main():
    spark = SparkSession.builder.getOrCreate()

    df = spark.table(SOURCE_TABLE)

    features = (
        df
        .withColumn(
            "total_spending",
            F.round(
                F.col("previous_purchases")
                * F.col("avg_purchase"),
                2,
            ),
        )
    )

    (
        features
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(FEATURE_TABLE)
    )

    print(f"Feature table updated: {FEATURE_TABLE}")


if __name__ == "__main__":
    main()