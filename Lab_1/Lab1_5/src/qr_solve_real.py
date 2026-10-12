import numpy as np


def offdiag_norm(matrix_A):
    buf = np.tril(matrix_A, k=-1)
    norm = np.sqrt(np.sum(buf ** 2))
    return norm


def find_eigenvalues(matrix_A):
    return np.diag(matrix_A)


def householder_vector(vector_b):
    m = vector_b.shape[0]
    b_norm = np.sqrt(np.sum(vector_b**2))
    e1 = np.eye(m)[0]
    sgn = 1.0 if vector_b[0] >= 0 else -1.0
    v = vector_b + sgn * b_norm * e1
    return v


def qr_decomposition(matrix_A):
    n = matrix_A.shape[0]
    R = matrix_A.copy()
    Q = np.eye(n)

    for k in range(n - 1):
        b = R[k:, k]
        v_small = householder_vector(b)
        v_full = np.zeros(n)
        v_full[k:] = v_small
        H = np.eye(n) - 2 * ((v_full[:, None] @ v_full[None, :]) / (v_full @ v_full))
        R = H @ R
        Q = Q @ H
    
    return Q, R


def qr_algorithm(matrix_A, eps, max_iter=1000):
    A_current = matrix_A.copy()
    
    for iteration in range(1, max_iter+1):
        norm = offdiag_norm(A_current)
        if norm <= eps:
            return A_current, iteration
        
        Q, R = qr_decomposition(A_current)
        A_current = R @ Q
    
    raise RuntimeError(f"QR-алгоритм не сошелся за {max_iter} итераций")


if __name__ == "__main__":
    A = np.array([[-2.,  7., -6.],
                  [-1.,  9., -4.],
                  [-1.,  8., -3.]])
    Q, R = qr_decomposition(A)
    print("Q = \n", Q)
    print("R = \n", R)

    check_QR = np.allclose(A, Q @ R)
    check_Q_orth = np.allclose(Q.T @ Q, np.eye(A.shape[0]))
    check_R_tri = np.allclose(R, np.triu(R))

    print("Проверка A = Q @ R:", check_QR)
    print("Проверка Q.T @ Q = E:", check_Q_orth)
    print("Проверка R верхнетреуг.:", check_R_tri)

    A_result, iterations = qr_algorithm(A, eps=1e-5)
    eigenvalues = find_eigenvalues(A_result)

    print("\nA_result = \n", A_result)
    print("Собственные значения = \n", eigenvalues)
    print("Число итераций =", iterations)
