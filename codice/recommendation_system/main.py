import os


os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"


import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf



from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator



conf = SparkConf().set("spark.ui.port", "4050")

sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()



schema_ratings = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("item_id", IntegerType(), False),
    StructField("rating", IntegerType(), False),
    StructField("timestamp", IntegerType(), False)
])

schema_items = StructType([
    StructField("item_id", IntegerType(), False),
    StructField("movie", StringType(), False)
])


training = spark.read.option("sep", "\t").csv(
    "data/u1.base",
    header=False,
    schema=schema_ratings
)




test = spark.read.option("sep", "\t").csv(
    "data/u1.test",
    header=False,
    schema=schema_ratings
)

items = spark.read.option("sep", "|").csv(
    "data/u.item",
    header=False,
    schema=schema_items
)



training.printSchema()
test.printSchema()
items.printSchema()

print("Prime 3 righe del training:")
print(training.take(3))

print("Prime 3 righe degli items:")
print(items.take(3))



reviews_per_movie = (
    training
    .groupBy("item_id")
    .count()
    .orderBy("item_id")
)


reviews_per_movie.show(5)


avg_rating_movie = (
    training
    .groupBy("item_id")
    .agg(avg("rating").alias("avg_rating"))
    .orderBy("item_id")
)
avg_rating_movie.show(5)



als = ALS(
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)


model = als.fit(training)


predictions = model.transform(test)

predictions.show()




evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)

print("RMSE = ", rmse)


user_recommendations = model.recommendForAllUsers(10)

user_recommendations.show(truncate=False)

recommendations = user_recommendations.select(
    "user_id",
    explode("recommendations").alias("recommendation")
)

recommendations = recommendations.select(
    "user_id",
    col("recommendation.item_id").alias("item_id"),
    col("recommendation.rating").alias("predicted_rating")
)

recommendations_with_movies = recommendations.join(
    items,
    on="item_id",
    how="left"
)

recommendations_with_movies.show(100, truncate=False)