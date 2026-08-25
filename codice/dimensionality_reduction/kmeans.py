import numpy as np

from pyspark.core.rdd import RDD
from pyspark.mllib.clustering import KMeans


def calculate_kmeans(vectors: RDD, k=2):

    model = KMeans.train(
        vectors,
        k=k,
        maxIterations=20
    )

    predictions = vectors.map(
        lambda point: (
            model.predict(point),
            point
        )
    )

    return model, predictions


def calculate_silhouette(predictions):

    data = predictions.collect()

    silhouettes = []

    for cluster_i, point_i in data:

        same_cluster = [
            point_j
            for cluster_j, point_j in data
            if cluster_j == cluster_i
        ]

        if len(same_cluster) > 1:

            distances_same = [
                np.linalg.norm(
                    point_i.toArray() - point_j.toArray()
                )
                for point_j in same_cluster
                if point_j is not point_i
            ]

            a = np.mean(distances_same)

        else:

            a = 0


        other_clusters = set(
            cluster_j
            for cluster_j, point_j in data
            if cluster_j != cluster_i
        )


        mean_distances = []

        for other_cluster in other_clusters:

            other_cluster_points = [
                point_j
                for cluster_j, point_j in data
                if cluster_j == other_cluster
            ]

            distances_other = [
                np.linalg.norm(
                    point_i.toArray() - point_j.toArray()
                )
                for point_j in other_cluster_points
            ]

            mean_distances.append(
                np.mean(distances_other)
            )


        b = min(mean_distances)


        if max(a, b) == 0:
            s = 0
        else:
            s = (b - a) / max(a, b)

        silhouettes.append(s)


    return np.mean(silhouettes)