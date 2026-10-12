import numpy as np
from utils import read_system_from_files
from lu_decomposition import lu_decompose
from solve import solve_system, inverse_matrix, determinant
from checks import check_lu, check_solution, check_inverse
from report_utils import add_section, add_matrix, add_value, add_list


MATRIX_A_PATH = "data/matrix_A.txt"
VECTOR_B_PATH = "data/vector_b.txt"
REPORT_PATH = "data/report_lab_1_1.txt"


def make_report(A, b, L, U, P, permutations, count, x, A_inv, det_A,
                res_LU, res_solution, res_inverse):
    lines = []
    lines.append("Лабораторная работа 1.1: LU-разложение")
    lines.append("")
    add_value(lines, "Размерность системы", A.shape[0])

    add_section(lines, "Входные данные")
    add_matrix(lines, "Матрица A", A)
    add_matrix(lines, "Вектор b", b)

    add_section(lines, "Выходные данные")
    add_matrix(lines, "L", L)
    add_matrix(lines, "U", U)
    add_matrix(lines, "L @ U", L @ U)
    add_value(lines, "Вектор перестановок", permutations)
    add_value(lines, "Число перестановок", count)
    add_matrix(lines, "Матрица P", P)
    add_matrix(lines, "Решение системы Ax = b", x)
    add_matrix(lines, "Обратная матрица A_inv", A_inv)
    add_value(lines, "Определитель det(A)", det_A)

    add_section(lines, "Проверки")
    lu_items = [f"{key} -> {value}" for key, value in res_LU.items()]
    add_list(lines, "LU-разложение", lu_items)
    add_value(lines, "Невязка системы", res_solution)
    add_value(lines, "Невязка обратной матрицы", res_inverse)

    return "\n".join(lines)


def main():
    A, b = read_system_from_files(MATRIX_A_PATH, VECTOR_B_PATH)
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