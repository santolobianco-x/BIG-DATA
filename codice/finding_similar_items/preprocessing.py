from pyspark.sql.functions import col, concat_ws, explode, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.linalg import Vectors, VectorUDT


def load_data(spark, train_path='data/train.csv', test_path='data/test.csv', train_len = 5000, test_len= 200):
    train = spark.read.option("header","false") \
        .option("inferSchema", "true")\
        .option("quote",'"') \
        .option("escape", '"') \
        .csv(train_path)


    test = spark.read\
        .option("header", "false") \
        .option("inferSchema", "true")\
        .option("quote",'"') \
        .option("escape", '"') \
        .csv(test_path)

    train = train.toDF("popolarity", "title", "text")
    test = test.toDF("popolarity", "title", "text")

    train = train.limit(train_len)
    test = test.limit(test_len)

    return train, test



def shingle(text, q=3):
    text = text.lower().replace(" ", "")

    return {
        text[i:i+q]
        for i in range(len(text)-q+1)
    }


def create_shingles(train, test):

    train = train.withColumn(
        "document",
        concat_ws(" ", col("title"), col("text"))
    )

    test = test.withColumn(
        "document",
        concat_ws(" ", col("title"), col("text"))
    )

    shingle_udf = udf(
        lambda text: list(shingle(text, 3)),
        ArrayType(StringType())
    )

    train = train.withColumn(
        "shingles",
        shingle_udf("document")
    )

    test = test.withColumn(
        "shingles",
        shingle_udf("document")
    )

    return train, test


def create_vocabulary(train):

    vocab = train \
        .select(explode("shingles").alias("shingle")) \
        .distinct() \
        .rdd \
        .map(lambda row: row["shingle"]) \
        .collect()

    shingle_to_index = {
        s: i for i, s in enumerate(vocab)
    }

    return vocab, shingle_to_index




def create_vectors(train, test, shingle_to_index, vocab_size):

    has_known_shingle_udf = udf(
        lambda shingles: any(s in shingle_to_index for s in shingles),
        "boolean"
    )

    test = test.filter(
        has_known_shingle_udf("shingles")
    )

    def create_sparse_vector(shingles):
        indices = sorted(
            set(
                shingle_to_index[s]
                for s in shingles
                if s in shingle_to_index
            )
        )

        values = [1.0] * len(indices)

        return Vectors.sparse(
            vocab_size, indices, values
        )    

    vector_udf = udf(create_sparse_vector, VectorUDT())

    train = train.withColumn("features", vector_udf("shingles"))
    test = test.withColumn("features", vector_udf("shingles"))
    return train, test