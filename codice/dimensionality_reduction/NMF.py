import os
os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS


spark = SparkSession.builder.appName("NMF_Sparse_PySpark").getOrCreate()


data = [
    (0, Vectors.sparse(100, {0: 1.0, 1: 2.0, 2: 3.0})),
     (1, Vectors.sparse(100, {1: 4.0, 10: 5.0, 20: 6.0})),
      (2, Vectors.sparse(100, {5: 7.0, 15: 8.0, 25: 9.0})),
       (3, Vectors.sparse(100, {0: 10.0, 10: 11.0, 30: 12.0})),
        (4, Vectors.sparse(100, {40: 13.0, 50: 14.0, 60: 15.0}))
]


columns = ["id", "features"]
df = spark.createDataFrame(data, columns)

data_als = df.rdd.flatMap(
    lambda row: [(row.id, i, float(v)) for i, v in enumerate(row.features.toArray()) if v > 0])

data_als_df = spark.createDataFrame(data_als, ["user", "item", "rating"])

als = ALS(
    maxIter=10,
    rank=2,
    regParam=0.1,
    userCol="user",
    itemCol="item",
    ratingCol="rating",
    nonnegative=True
)

model = als.fit(data_als_df)
user_factors = model.userFactors
item_factors = model.itemFactors


print("User Factors:")
user_factors.show(truncate=False)

print("Item Factors:")
item_factors.show(truncate=False)

spark.stop()