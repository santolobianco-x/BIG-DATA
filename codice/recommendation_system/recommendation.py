from pyspark.sql.functions import avg, explode, col
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame



def calculate_statistics(training: DataFrame):
    reviews_per_movie = (
        training
        .groupBy("item_id")
        .count()
        .orderBy("item_id")
    )

    avg_rating_movie = (
        training
        .groupBy("item_id")
        .agg(avg("rating").alias("avg_rating"))
        .orderBy("item_id")
    )

    return reviews_per_movie, avg_rating_movie


def train_model(training: DataFrame):
    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )

    model = als.fit(training)

    return model


def calculate_rmse(model: ALSModel, test: DataFrame):
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    rmse = evaluator.evaluate(predictions)
    return predictions, rmse

def generate_recommendations(model: ALSModel, items: DataFrame, k=10):

    user_recommendations = model.recommendForAllUsers(k)

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

    return recommendations_with_movies