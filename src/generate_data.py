from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
)

import random
from datetime import datetime


CATALOG = "main"
SCHEMA = "ml_demo"
TABLE = f"{CATALOG}.{SCHEMA}.customer_data"


def create_database(spark: SparkSession):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")


def generate_rows(start_id: int, count: int):
    rows = []

    for customer_id in range(start_id, start_id + count):
        age = random.randint(20, 60)
        income = random.randint(30000, 90000)
        previous_purchases = random.randint(0, 10)
        avg_purchase = round(random.uniform(20, 150), 2)

        # Very simple artificial target
        will_buy = int(
            previous_purchases >= 4
            and avg_purchase >= 60
        )

        rows.append(
            (
                customer_id,
                age,
                income,
                previous_purchases,
                avg_purchase,
                will_buy,
                datetime.utcnow(),
            )
        )

    return rows


def main():
    spark = SparkSession.builder.getOrCreate()

    create_database(spark)

    table_exists = spark.catalog.tableExists(TABLE)

    if not table_exists:
        count = 50
        start_id = 1

        print("Creating initial dataset with 50 rows.")

    else:
        max_id = (
            spark.table(TABLE)
            .select(F.max("customer_id"))
            .collect()[0][0]
        )

        start_id = max_id + 1
        count = 2

        print(f"Adding {count} new rows starting from customer_id={start_id}")

    rows = generate_rows(start_id, count)

    schema = StructType(
        [
            StructField("customer_id", IntegerType(), False),
            StructField("age", IntegerType(), False),
            StructField("income", IntegerType(), False),
            StructField("previous_purchases", IntegerType(), False),
            StructField("avg_purchase", DoubleType(), False),
            StructField("will_buy", IntegerType(), False),
            StructField("created_at", "timestamp", False),
        ]
    )

    df = spark.createDataFrame(rows, schema)

    if not table_exists:
        df.write.format("delta").saveAsTable(TABLE)
    else:
        df.write.format("delta").mode("append").saveAsTable(TABLE)

    print(f"Added {count} rows.")
    print(f"Total rows: {spark.table(TABLE).count()}")


if __name__ == "__main__":
    main()