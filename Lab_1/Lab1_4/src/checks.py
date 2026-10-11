import numpy as np


def check_AV_VL(matrix_A, matrix_V, eigenvalues):
    Lambda = np.diag(eigenvalues)
    return np.allclose(matrix_A @ matrix_V, matrix_V @ Lambda)


def check_orthogonality(matrix):
    n = matrix.shape[0]
    return np.allclose(matrix.T @ matrix, np.eye(n))

    
if __name__ == "__main__":
    from jacobi_rotation import jacobi_rotation_solve
    A = np.array([[ 2.,  8.,  7.],
                  [ 8.,  2.,  7.],
                  [ 7.,  7., -8.]])
    eigenvalues, V, iterations = jacobi_rotation_solve(A, eps=0.001)
    print("check AV = VΛ:", check_AV_VL(A, V, eigenvalues))
    print("check orthogonality:", check_orthogonality(V))