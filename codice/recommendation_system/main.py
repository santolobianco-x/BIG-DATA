import os

os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from preprocessing import load_data
from recommendation import calculate_statistics, train_model, calculate_rmse, generate_recommendations


conf = SparkConf().set("spark.ui.port", "4050")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

DATA_PATH = "data"


training, test, items = load_data(spark, DATA_PATH)



reviews_per_movie, avg_rating_movie = calculate_statistics(training)

print("\nNumero di recensioni per film:")
reviews_per_movie.show(5)

print("\nRating medio per film:")
avg_rating_movie.show(5)



model = train_model(training)



predictions, rmse = calculate_rmse(model, test)

print("\nPredizioni:")
predictions.show(5)

print("\nRMSE =", rmse)



recommendations = generate_recommendations(model, items, k=10)

print("\nTop 10 raccomandazioni per ogni utente:")
recommendations.show(100, truncate=False)

spark.stop()