from sklearn.datasets import load_breast_cancer
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import array
from pyspark.mllib.linalg import Vectors

def load_data(spark: SparkSession):
    breast_cancer = load_breast_cancer()

    pd_df = pd.DataFrame(
        breast_cancer.data,
        columns= breast_cancer.feature_names
    )


    df = spark.createDataFrame(pd_df)




    for struct_field in df.schema:
        struct_field.nullable = False

    df = spark.createDataFrame(df.rdd, df.schema)

    df = df.withColumn(
        "features",
        array(*df.columns)
    )

    vectors = df.rdd.map(
        lambda row: Vectors.dense(row.features)
    )


    labels = breast_cancer.target


    return vectors, labels, breast_cancer.feature_names