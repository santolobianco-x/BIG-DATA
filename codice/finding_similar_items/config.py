import os
os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"


import pyspark
from pyspark.sql import SparkSession



conf = pyspark.SparkConf().set("spark.ui.port", "4050")

sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()