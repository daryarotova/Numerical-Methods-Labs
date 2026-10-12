import numpy as np
import cmath
from qr_solve_real import qr_decomposition


def eigenvalues_of_block(A, j):
    a = A[j, j]
    b = A[j, j+1]
    c = A[j+1, j]
    d = A[j+1, j+1]
    trace = a + d
    determ = a * d - b * c
    discr = trace ** 2 - 4 * determ
    lam1 = (trace + cmath.sqrt(discr)) / 2
    lam2 = (trace - cmath.sqrt(discr)) / 2
    return lam1, lam2


def find_complex_block(A):
    n = A.shape[0]
    limit = 1e-10
    
    for j in range(n - 1):
        lam1, lam2 = eigenvalues_of_block(A, j)
        
        if abs(lam1.imag) > limit or abs(lam2.imag) > limit:
            return j
    
    return None


def qr_complex_algorithm(matrix_A, eps, max_iter=1000):
    A_current = matrix_A.copy()
    n = A_current.shape[0]

    j = find_complex_block(A_current)
    if j is None:
        raise ValueError("В матрице нет комплексной пары значений, нужно использовать обычный QR-алгоритм")
    
    lam_prev = eigenvalues_of_block(A_current, j)

    for iteration in range(1, max_iter + 1):
        Q, R = qr_decomposition(A_current)
        A_current = R @ Q
        j_new = find_complex_block(A_current)
        if j_new is None:
            raise RuntimeError("Блок исчез, метод невозможно применить")
        lam_new = eigenvalues_of_block(A_current, j_new)
        diff1 = abs(lam_new[0] - lam_prev[0])
        diff2 = abs(lam_new[1] - lam_prev[1])
        if max(diff1, diff2) < eps:
            return A_current, iteration
        lam_prev = lam_new
    
    raise RuntimeError(f"QR-алгоритм не сошелся за {max_iter} итераций")


def extract_all_eigenvalues(A_final):
    n = A_final.shape[0]
    eigenvalues = []
    i = 0
    while i < n:
        if i < n - 1:
            lam1, lam2 = eigenvalues_of_block(A_final, i)
            if abs(lam1.imag) > 1e-10 or abs(lam2.imag) > 1e-10:
                eigenvalues.append(complex(lam1))
                eigenvalues.append(complex(lam2))
                i += 2
                continue
        
        eigenvalues.append(float(A_final[i, i]))
        i += 1
    
    return eigenvalues


if __name__ == "__main__":
    B = np.array([[2., 3., 1.],
                  [-3.,  1., 1.],
                  [1.,  1., 1.]])
    print("Матрица B =\n", B)

    lam1, lam2 = eigenvalues_of_block(B, 0)
    print("Собственные значения блока B[0:2, 0:2]:", lam1, lam2)

    j = find_complex_block(B)
    print("Индекс комплексного блока:", j)

    B_final, iter_B = qr_complex_algorithm(B, eps=1e-6)
    print("\nB_final =\n", B_final)
    print("Число итераций:", iter_B)

    j_new = find_complex_block(B_final)
    lam1, lam2 = eigenvalues_of_block(B_final, j_new)
    print("Комплексно-сопряжённыe cобственные значения блока:\n", lam1, lam2)

    all_eig = extract_all_eigenvalues(B_final)
    print("\nВсе собственные значения:\n", all_eig)
    print("Сумма ~ след:", np.allclose(np.sum(all_eig), np.trace(B)))

    print("Проверка через numpy:", np.linalg.eigvals(B))