import numpy as np


def lu_decompose(matrix_A):
    if matrix_A.ndim != 2 or matrix_A.shape[0] != matrix_A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    U = matrix_A.copy()
    n = matrix_A.shape[0]
    permutations = np.arange(n)
    count_of_permutations = 0
    L = np.eye(n, dtype=float)

    for k in range(0, n - 1):
        s = np.argmax(np.abs(U[k:, k])) + k

        if s != k:
            U[[k, s]] = U[[s, k]]
            L[[k, s], :k] = L[[s, k], :k]
            permutations[k], permutations[s] = permutations[s], permutations[k]
            count_of_permutations += 1
        
        for i in range(k + 1, n):
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            
            for j in range(k, n):
                U[i, j] -= multiplier * U[k, j]

    return L, U, permutations, count_of_permutations


if __name__ == "__main__":    
    A = np.array([[0.0, 2.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [2.0, 3.0, 0.0]])
    
    L, U, perm, count = lu_decompose(A)
    n = A.shape[0]
    P = np.eye(n)[perm, :]

    print("L @ U: \n", L @ U)
    print("P @ A: \n", P @ A)

    check_A_and_LU = np.allclose(A, L @ U)
    check_L = np.allclose(L, np.tril(L))
    check_U = np.allclose(U, np.triu(U))
    check_diag = np.allclose(np.diag(L), np.ones(n))
    checkPAandLU = np.allclose(P @ A, L @ U)

    print(
        "\n1. A = LU", check_A_and_LU, # False
        "\n2. L", check_L,
        "\n3. U", check_U,
        "\n4. diag", check_diag,
        "\n5. PA = LU", checkPAandLU # True
    )

