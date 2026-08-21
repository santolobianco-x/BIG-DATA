import builtins
from collections import Counter


def predict(model, train_hashed, test_hashed, k=3):

    risultati = []

    for row in test_hashed.collect():

        key = row["features"]

        neighbors = model.approxNearestNeighbors(
            train_hashed,
            key,
            3,
            distCol="JaccardDistance"
        )

        popolarities = [
            neighbor["popolarity"]
            for neighbor in neighbors.select("popolarity").collect()
        ]

        if len(popolarities) > 0:
            prediction = Counter(popolarities).most_common(1)[0][0]
        else:
            prediction = None

        risultati.append(
            (row["popolarity"], prediction)
        )
    return risultati


def calculate_accuracy(risultati):
    corrette = builtins.sum(
        1
        for reale, predetta in risultati
        if predetta is not None and reale == predetta
    )

    totale = builtins.sum(
        1
        for reale, predetta in risultati
        if predetta is not None
    )

    accuracy = corrette / totale if totale > 0 else 0


    return corrette, totale, accuracy