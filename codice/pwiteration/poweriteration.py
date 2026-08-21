import numpy as np
from pyspark import RDD
from matrix_vector import matrix_vector_multiply



def power_iteration(matrix_rdd: RDD, num_nodes: int, initial_vector, sc, max_iterations: int =20, convergence_threshold: float = 1e-6) -> np.ndarray:
    vector = np.array(
        [value for index, value in sorted(initial_vector)],
        dtype=float
    )

    vector = vector / np.linalg.norm(vector)

    vector_broadcast = sc.broadcast(vector)

    matrix_by_row = (
        matrix_rdd
        .map(lambda x: (x[0], (x[1], x[2])))
        .groupByKey()
        .mapValues(list)
    )

    iteration = 0
    converged = False

    prev_vector = vector.copy()
    # si salva il vettore precedente per controllare la convergenza
    
    while iteration < max_iterations and not converged:

        #calcoliamo A*v
        result_rdd = matrix_vector_multiply(
            matrix_by_row,
            vector_broadcast
        )

        new_vector = np.zeros(num_nodes)

        for row, value in result_rdd.collect():
            new_vector[row] = value

        norm = np.linalg.norm(new_vector)

        if norm != 0:
            new_vector = new_vector / norm

        difference = np.linalg.norm(
            new_vector - prev_vector
        )

        if difference < convergence_threshold:
            converged = True

        prev_vector = new_vector.copy()
        vector = new_vector

        #si aggiorna il broadcast con il nuovo vettore
        vector_broadcast.unpersist()
        vector_broadcast = sc.broadcast(vector)

        iteration += 1

        print(
            f"Iterazione {iteration}: "
            f"differenza = {difference:.8f}"
        )

    vector_broadcast.unpersist()

    return vector

