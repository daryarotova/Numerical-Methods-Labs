import numpy as np


def check_qr_decomposition(matrix_A, matrix_Q, matrix_R):
    return np.allclose(matrix_A, matrix_Q @ matrix_R)


def check_orthogonality(matrix_Q):
    n = matrix_Q.shape[0]
    return np.allclose(matrix_Q.T @ matrix_Q, np.eye(n))


def check_upper_triangular(matrix_R):
    return np.allclose(matrix_R, np.triu(matrix_R))


def check_eigenvalues_sum(matrix_A, eigenvalues):
    return np.allclose(np.sum(eigenvalues), np.trace(matrix_A))


def check_eigenvalues_product(matrix_A, eigenvalues):
    return np.allclose(np.prod(eigenvalues), np.linalg.det(matrix_A))


if __name__ == "__main__":
    from qr_solve_real import qr_decomposition, qr_algorithm, find_eigenvalues

    A = np.array([[-2., 7., -6.],
                  [-1., 9., -4.],
                  [-1., 8., -3.]])

    Q, R = qr_decomposition(A)
    print("A = Q @ R:", check_qr_decomposition(A, Q, R))
    print("Q orthogonal:", check_orthogonality(Q))
    print("R upper triangular:", check_upper_triangular(R))

    A_result, iterations = qr_algorithm(A, eps=1e-6)
    eigenvalues = find_eigenvalues(A_result)
    print("Sum of eigenvalues = trace:", check_eigenvalues_sum(A, eigenvalues))
    print("Product of eigenvalues = determinant:", check_eigenvalues_product(A, eigenvalues))
