import numpy as np


def check_lu(matrix_A, matrix_L, matrix_U, permutations):
    res_APLU = np.allclose(matrix_A[permutations, :], matrix_L @ matrix_U)
    res_L = np.allclose(matrix_L, np.tril(matrix_L))
    res_U = np.allclose(matrix_U, np.triu(matrix_U))
    res_L_diag = np.allclose(np.diag(matrix_L), np.ones(matrix_A.shape[0]))

    result = {
        "P * A = L * U": res_APLU,
        "L is lower triangle matrix": res_L,
        "U is upper triangle matrix": res_U,
        "L diag = 1": res_L_diag
    }
    return result


def check_solution(matrix_A, vector_b, vector_x):
    residual = np.linalg.norm(matrix_A @ vector_x - vector_b)
    return residual


def check_inverse(matrix_A, matrix_A_inv):
    E = np.eye(matrix_A.shape[0])
    residual = np.linalg.norm(matrix_A @ matrix_A_inv - E)
    return residual


if __name__ == '__main__':
    from lu_decomposition import lu_decompose
    from solve import solve_system, inverse_matrix

    A = np.array([[2.0, 1.0, 1.0],
                  [4.0, -6.0, 0.0],
                  [-2.0, 7.0, 2.0]])
    x_true = np.array([1.0, 2.0, 3.0])
    b = A @ x_true

    L, U, perm, _ = lu_decompose(A)
    print("check_lu:")
    for key, value in check_lu(A, L, U, perm).items():
        print(f"  {key} -> {value}")

    x = solve_system(A, b)
    print("check_solution: ", check_solution(A, b, x))

    A_inv = inverse_matrix(A)
    print("check_inverse:  ", check_inverse(A, A_inv))