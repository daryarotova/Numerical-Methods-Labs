import numpy as np
import math


def find_max_off_diag(matrix_A):
    maximum = 0
    n = matrix_A.shape[0]
    i_res, j_res = 0, 1

    for i in range(n):
        for j in range(i+1, n):
            value = np.abs(matrix_A[i, j])
            if value > maximum:
                maximum = value
                i_res, j_res = i, j

    return i_res, j_res, maximum


def build_rotation_matrix(n, i, j, phi):
    matrix_U = np.eye(n)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    matrix_U[i, i] = cos_phi
    matrix_U[i, j] = -sin_phi
    matrix_U[j, i] = sin_phi
    matrix_U[j, j] = cos_phi
    return matrix_U


def off_diag_norm(matrix_A):
    upper = np.triu(matrix_A, k=1)
    return np.sqrt(np.sum(upper ** 2))


def jacobi_rotation_solve(matrix_A, eps, max_iter=1000):
    n = matrix_A.shape[0]
    A_current = matrix_A.copy()
    V = np.eye(n)

    for iteration in range(1, max_iter+1):
        norm = off_diag_norm(A_current)
        if norm < eps:
            eigenvalues = np.diag(A_current)
            return eigenvalues, V, iteration
        
        i, j, _ = find_max_off_diag(A_current)
        angle_phi = 0.5 * np.arctan2(2 * A_current[i, j], A_current[i, i] - A_current[j, j])

        U = build_rotation_matrix(n, i, j, angle_phi)
        A_current = U.T @ A_current @ U
        V = V @ U
    
    raise RuntimeError(f"Метод вращений Якоби не сошелся за {max_iter} итераций")


if __name__ == "__main__":
    A = np.array([[ 2.,  8.,  7.],
                [ 8.,  2.,  7.],
                [ 7.,  7., -8.]])

    i, j, val = find_max_off_diag(A)
    print("maximum off diag:", i, j, val)

    U = build_rotation_matrix(3, 0, 1, 0.5)
    print("\nU =\n", U)
    print("U.T @ U =\n", U.T @ U)
    
    print("norm =", off_diag_norm(A))

    eigenvalues, V, iterations = jacobi_rotation_solve(A, eps=0.001)
    print("\neigenvalues =\n", eigenvalues)
    print("V =\n", V)
    print("iterations =", iterations)

    Lambda = np.diag(eigenvalues)
    print("A @ V = V @ Lambda:", np.allclose(A @ V, V @ Lambda))
    print("V.T @ V = E:", np.allclose(V.T @ V, np.eye(3)))