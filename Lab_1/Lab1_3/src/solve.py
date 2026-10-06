import numpy as np


def build_equivalent_form(matrix_A, vector_b):
    d = np.diag(matrix_A)
    if np.any(np.abs(d) < 1e-15):
        raise ValueError(
            "На главной диагонали матрицы A есть нули => "
            "приведение невозможно без перестановки строк"
        )
    beta = vector_b / d
    alpha = - matrix_A / d[:, None]
    np.fill_diagonal(alpha, 0)
    return alpha, beta


def matrix_norm_c(matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))


def vector_norm_c(vector):
    return np.max(np.abs(vector))


def jacobi_solve(alpha, beta, eps, max_iter=1000):
    norm_alpha = matrix_norm_c(alpha)
    if norm_alpha >= 1:
        raise RuntimeError(
            f"Достаточное условие сходимости не выполняется для метода Якоби: "
            f"||alpha||_c = {norm_alpha} >= 1"
        )

    x = beta.copy()
    for k in range(1, max_iter+1):
        x_new = beta + alpha @ x
        diff = vector_norm_c(x_new - x)

        eps_k = (norm_alpha / (1 - norm_alpha)) * diff

        if eps_k <= eps:
            return x_new, k
        x = x_new
    
    raise RuntimeError(f"Метод Якоби не сошёлся за {max_iter} итераций")


def seidel_solve(alpha, beta, eps, max_iter=1000):
    norm_alpha = matrix_norm_c(alpha)
    if norm_alpha >= 1:
        raise RuntimeError(
            f"Достаточное условие сходимости не выполняется для метода Зейделя: "
            f"||alpha||_c = {norm_alpha} >= 1"
        )
    
    norm_C = matrix_norm_c(np.triu(alpha, k=1))
    n = len(beta)
    x = beta.copy()

    for k in range(1, max_iter+1):
        x_prev = x.copy()
        for row_index in range(n):
            x_new_value = beta[row_index]
            for col_index in range(n):
                if col_index != row_index:
                    x_new_value += alpha[row_index, col_index] * x[col_index]
            x[row_index] = x_new_value

        diff = vector_norm_c(x - x_prev)
        eps_k = (norm_C / (1 - norm_alpha)) * diff

        if eps_k <= eps:
            return x, k

    raise RuntimeError(f"Метод Зейделя не сошёлся за {max_iter} итераций")


if __name__ == "__main__":
    A = np.array([[10., 1., 1.],
                  [2., 10., 1.],
                  [2., 2., 10.]])
    b = np.array([12., 13., 14.])

    alpha, beta = build_equivalent_form(A, b)
    print("alpha =\n", alpha)
    print("beta =", beta)
    print("||alpha||_c =", matrix_norm_c(alpha))

    x_j, k_j = jacobi_solve(alpha, beta, eps=0.01)
    print("Jacobi: \n x =", x_j)
    print(" k =", k_j)

    x_s, k_s = seidel_solve(alpha, beta, eps=0.01)
    print("Seidel: \n x =", x_s)
    print(" k =", k_s)