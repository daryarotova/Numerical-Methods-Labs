import numpy as np
from lu_decomposition import lu_decompose


def solve_lower(matrix_L, vector_b):
    n = matrix_L.shape[0]
    z = np.zeros(n)

    for i in range(n):
        z[i] = vector_b[i]
        for j in range(0, i):
            z[i] -= matrix_L[i, j] * z[j]
    
    return z


def solve_upper(matrix_U, vector_b):
    n = matrix_U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = vector_b[i]
        for j in range(i + 1, n):
            x[i] -= matrix_U[i, j] * x[j]
        x[i] /= matrix_U[i, i]
    
    return x


def solve_system(matrix_A, vector_b):
    L, U, permutations, _ = lu_decompose(matrix_A)
    Pb = vector_b[permutations]
    z = solve_lower(L, Pb)
    x = solve_upper(U, z)
    return x


def inverse_matrix(matrix_A):
    L, U, permutations, _ = lu_decompose(matrix_A)
    n = matrix_A.shape[0]
    E = np.eye(n)
    A_inv = np.zeros((n, n))

    for k in range(n):
        e_k = E[:, k] # k-й столбец E — это вектор с 1 на позиции k
        P_e_k = e_k[permutations] # перестановка
        z = solve_lower(L, P_e_k)
        x_k = solve_upper(U, z)
        A_inv[:, k] = x_k # записываем в k-ый столбец рез-та
    
    return A_inv


def determinant(matrix_A):
    L, U, permutations, count = lu_decompose(matrix_A)
    n = matrix_A.shape[0]
    det_U = 1.0
    for i in range(n):
        det_U *= U[i, i]
    
    det_A = ((-1) ** count) * det_U
    return det_A


if __name__ == '__main__':
    A = np.array([[2.0, 1.0, 1.0],
                  [4.0, -6.0, 0.0],
                  [-2.0, 7.0, 2.0]])
    x_true = np.array([1.0, 2.0, 3.0])
    b = A @ x_true

    x = solve_system(A, b)
    check_system = np.allclose(x, x_true)

    A_inv = inverse_matrix(A)
    check_inverse = np.allclose(A @ A_inv, np.eye(A.shape[0]))

    det_A = determinant(A)
    det_A_np = np.linalg.det(A)
    check_det = np.allclose(det_A, det_A_np)

    print("1. solve_system: ", check_system)
    print("2. inverse_matrix: ", check_inverse)
    print("3. determinant: ", check_det)