from pyspark.ml.feature import MinHashLSH

def train_minhash(train):
    mh = MinHashLSH(
        inputCol="features",
        outputCol="hashes",
        numHashTables=5
    )

    model = mh.fit(train)
    return model


def hash_data(model, train, test):
    train_hashed = model.transform(train)
    test_hashed = model.transform(test)
    return train_hashed, test_hashed