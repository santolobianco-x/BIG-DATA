import os
import time

from pyspark import SparkContext

from poweriteration import power_iteration


os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python"


matrix_entries = [
    (0, 0, 1),
    (0, 2, 3),
    (1, 0, 4),
    (1, 1, 5),
    (1, 2, 6),
    (2, 0, 7),
    (2, 1, 8),
    (2, 2, 9)
]

vector_entries = [
    (0, 10),
    (1, 20),
    (2, 30)
]


def main():

    sc = SparkContext("local", "Power Iteration")

    matrix_rdd = sc.parallelize(matrix_entries)

    eigenvector = power_iteration(
        matrix_rdd,
        num_nodes=3,
        initial_vector=vector_entries,
        sc=sc,
        max_iterations=50
    )

    print("\nAutovettore dominante:")

    for i, value in enumerate(eigenvector):
        print(f"Elemento {i}: {value:.6f}")

    sc.stop()


if __name__ == "__main__":
    main()