

from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import SparkSession




def load_data(spark: SparkSession, data_path: str):
    
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
        f"{data_path}/u1.base",
        header=False,
        schema=schema_ratings
    )




    test = spark.read.option("sep", "\t").csv(
        f"{data_path}/u1.test",
        header=False,
        schema=schema_ratings
    )

    items = spark.read.option("sep", "|").csv(
        f"{data_path}/u.item",
        header=False,
        schema=schema_items
    )

    return training, test, items