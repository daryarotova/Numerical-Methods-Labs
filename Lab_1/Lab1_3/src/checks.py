import numpy as np


def check_solution(matrix_A, vector_b, vector_x):
    residual = np.max(np.abs(matrix_A @ vector_x - vector_b))
    return residual


def solve_reference(matrix_A, vector_b):
    return np.linalg.solve(matrix_A, vector_b)


def check_convergence_condition(alpha):
    norm_alpha_c = np.max(np.sum(np.abs(alpha), axis=1))
    return norm_alpha_c, norm_alpha_c < 1


def check_diagonal_dominance(matrix_A):
    diag_abs = np.abs(np.diag(matrix_A))
    sum_non_diag = np.sum(np.abs(matrix_A), axis=1) - diag_abs
    all_non_strict = np.all(diag_abs >= sum_non_diag)
    any_strict = np.any(diag_abs > sum_non_diag)
    return all_non_strict and any_strict


if __name__ == "__main__":
    A = np.array([[10., 1., 1.],
                  [2., 10., 1.],
                  [2., 2., 10.]])
    b = np.array([12., 13., 14.])

    x_ref = solve_reference(A, b)
    print("x_ref =", x_ref)

    res = check_solution(A, b, x_ref)
    print("residual for x_ref =", res)

    alpha = -A / np.diag(A)[:, None]
    np.fill_diagonal(alpha, 0)
    norm_alpha, converged = check_convergence_condition(alpha)
    print("||alpha||_c =", norm_alpha, "->", converged)

    print("diagonal dominance:", check_diagonal_dominance(A))