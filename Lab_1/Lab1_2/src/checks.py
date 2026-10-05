import numpy as np
from sweep import sweep_backward, sweep_forward


def check_stability(a, b, c):
    left = np.abs(b)
    right = np.abs(a) + np.abs(c)

    all_non_strict = np.all(left >= right)
    any_strict = np.any(left > right)
    return all_non_strict and any_strict


def check_sweep_solution(a, b, c, d, vector_x):
    n = b.shape[0]
    matrix_A = (np.diag(b) + np.diag(a[1:n], k=-1) + np.diag(c[:n-1], k=1))
    residual = np.linalg.norm(matrix_A @ vector_x - d)
    return residual


if __name__ == '__main__':
    a = np.array([0., -1., 2., -1.])
    b = np.array([8., 6., 10., 6.])
    c = np.array([-2., -2., -4., 0.])
    d = np.array([6., 3., 8., 5.])

    P, Q = sweep_forward(a, b, c, d)
    x = sweep_backward(P, Q)
    
    res_stability = check_stability(a, b, c)
    res_solution = check_sweep_solution(a, b, c, d, x)
    print("check stability: ", res_stability)
    print("check solution - residual: ", res_solution)
