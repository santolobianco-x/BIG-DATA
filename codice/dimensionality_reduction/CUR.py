import numpy as np


def cur_decomposition(A, rank):

    m, n = A.shape

    total_norm = np.sum(A ** 2)

    col_probs = np.sum(A ** 2, axis=0) / total_norm
    row_probs = np.sum(A ** 2, axis=1) / total_norm

    col_indices = np.random.choice(
        n, rank, replace=False, p=col_probs
    )

    row_indices = np.random.choice(
        m, rank, replace=False, p=row_probs
    )

    C = A[:, col_indices]
    R = A[row_indices, :]

    W = A[np.ix_(row_indices, col_indices)]

    U = np.linalg.pinv(W)

    return C, U, R