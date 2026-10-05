import numpy as np
from utils import read_system_from_files
from lu_decomposition import lu_decompose
from solve import solve_system, inverse_matrix, determinant
from checks import check_lu, check_solution, check_inverse


REPORT_PATH = "data/report_lab_1_1.txt"


def make_report(A, b, L, U, P, permutations, count, x, A_inv, det_A, 
                res_LU, res_solution, res_inverse):
    lines = []
    lines.append("Лабораторная работа 1.1: LU-разложение")
    lines.append("")
    lines.append(f"Размерность системы: n = {A.shape[0]}")
    lines.append("")

    lines.append("--- Входные данные: ---")
    lines.append("Матрица A =")
    lines.append(str(A))
    lines.append("")

    lines.append("Вектор b =")
    lines.append(str(b))
    lines.append("")

    lines.append("--- Выходные данные: ---")
    lines.append("L =")
    lines.append(str(L))
    lines.append("")

    lines.append("U =")
    lines.append(str(U))
    lines.append("")

    lines.append("L @ U =")
    lines.append(str(L @ U))
    lines.append("")

    lines.append(f"Вектор перестановок: {permutations}")
    lines.append(f"Число перестановок: {count}")
    lines.append("")

    lines.append("Матрица P =")
    lines.append(str(P))
    lines.append("")

    lines.append("Решение системы Ax = b:")
    lines.append(str(x))
    lines.append("")

    lines.append("Обратная матрица A_inv =")
    lines.append(str(A_inv))
    lines.append("")

    lines.append(f"Определитель det(A) = {det_A}")
    lines.append("")

    lines.append("--- Проверки: ---")
    lines.append("LU-разложение:")
    for key, value in res_LU.items():
        lines.append(f"    {key} -> {value}")
    lines.append("")
    lines.append(f"Невязка системы: {res_solution}")
    lines.append("")
    lines.append(f"Невязка обратной матрицы: {res_inverse}")

    return "\n".join(lines)


def main():
    A, b = read_system_from_files("data/matrix_A.txt", "data/vector_b.txt")
    L, U, permutations, count = lu_decompose(A)
    P = np.eye(A.shape[0])[permutations, :]

    x = solve_system(A, b)
    A_inv = inverse_matrix(A)
    det_A = determinant(A)

    res_LU = check_lu(A, L, U, permutations)
    res_solution = check_solution(A, b, x)
    res_inverse = check_inverse(A, A_inv)

    report = make_report(A, b, L, U, P, permutations, count, x, A_inv, det_A, 
                         res_LU, res_solution, res_inverse)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print("Отчет записан в файл:", REPORT_PATH)


if __name__ == "__main__":
    main()