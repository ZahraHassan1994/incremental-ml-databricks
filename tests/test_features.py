from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def test_total_spending():

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("test")
        .getOrCreate()
    )

    data = [
        (1, 5, 80.0),
        (2, 2, 30.0),
    ]

    df = spark.createDataFrame(
        data,
        [
            "customer_id",
            "previous_purchases",
            "avg_purchase",
        ],
    )

    result = df.withColumn(
        "total_spending",
        F.col("previous_purchases")
        * F.col("avg_purchase"),
    )

    values = [
        row.total_spending
        for row in result.collect()
    ]

    assert values == [400.0, 60.0]

    spark.stop()