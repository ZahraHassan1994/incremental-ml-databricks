from pyspark.sql import SparkSession

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


CATALOG = "main"
SCHEMA = "ml_demo"

FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.customer_purchase_model"


FEATURE_COLUMNS = [
    "age",
    "income",
    "previous_purchases",
    "avg_purchase",
    "total_spending",
]

TARGET_COLUMN = "will_buy"


def main():
    spark = SparkSession.builder.getOrCreate()

    df = (
        spark.table(FEATURE_TABLE)
        .select(FEATURE_COLUMNS + [TARGET_COLUMN])
        .toPandas()
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
    )

    mlflow.set_registry_uri("databricks-uc")

    with mlflow.start_run() as run:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        mlflow.log_param(
            "n_estimators",
            50,
        )

        mlflow.log_param(
            "max_depth",
            4,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Model registered: {MODEL_NAME}")
        print(f"Accuracy: {accuracy}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()