import numpy as np
from checks import check_eigenvalues_product, check_orthogonality, check_eigenvalues_sum, check_qr_decomposition, check_upper_triangular
from utils import read_qr_system_from_files
from qr_solve_real import qr_algorithm, qr_decomposition, find_eigenvalues
from qr_solve_complex import qr_complex_algorithm, extract_all_eigenvalues
from report_utils import add_section, add_matrix, add_value, add_list

MATRIX_A_PATH = "data/matrix_A.txt"
MATRIX_B_PATH = "data/matrix_B_complex.txt"
EPSILON_PATH = "data/epsilon.txt"
REPORT_PATH = "data/report_lab_1_5.txt"


def sort_complex(values):
    arr = np.array(values, dtype=complex)
    return arr[np.lexsort((arr.imag, arr.real))]


def make_report(A, eps_A, Q, R, A_final, iterations, eigenvalues_A,
                res_QR, res_orth, res_tri, res_sum, res_prod,
                B, B_final, iteration_B, eigenvalues_B, res_B):
    lines = []
    lines.append("Лабораторная работа 1.5: QR-алгоритм")
    lines.append("")
    add_value(lines, "Размерность системы", A.shape[0])

    add_section(lines, "Входные данные")
    add_matrix(lines, "Матрица A", A)
    add_value(lines, "Точность epsilon", eps_A)

    add_section(lines, "QR-разложение")
    add_matrix(lines, "Матрица Q", Q)
    add_matrix(lines, "Матрица R", R)
    add_matrix(lines, "Результат умножения Q @ R", Q @ R)

    add_section(lines, "Проверки QR-разложения")
    add_value(lines, "Проверка A = Q @ R", res_QR)
    add_value(lines, "Проверка ортогональности Q", res_orth)
    add_value(lines, "Проверка верхней треугольности R", res_tri)

    add_section(lines, "QR-алгоритм")
    add_matrix(lines, "Финальная матрица A_final", A_final)
    add_value(lines, "Собственные значения", eigenvalues_A)
    add_value(lines, "Число итераций", iterations)

    add_section(lines, "Проверки собственных значений")
    add_value(lines, "Проверка суммы собственных значений и следа A", res_sum)
    add_value(lines, "Проверка произведения собственных значений и det A", res_prod)

    add_section(lines, "Пример с комплексно-сопряжёнными собственными значениями")
    add_matrix(lines, "Матрица B", B)
    add_matrix(lines, "Финальная матрица B_final", B_final)
    add_value(lines, "Число итераций", iteration_B)
    add_value(lines, "Собственные значения", eigenvalues_B)
    add_value(lines, "Собственные значения через numpy", np.linalg.eigvals(B))
    add_value(lines, "Проверка собственных значений", res_B)

    return "\n".join(lines)


def main():
    A, eps_A = read_qr_system_from_files(MATRIX_A_PATH, EPSILON_PATH)
    Q, R = qr_decomposition(A)
    
    res_QR = check_qr_decomposition(A, Q, R)
    res_orth = check_orthogonality(Q)
    res_tri = check_upper_triangular(R)

    A_final, iterations = qr_algorithm(A, eps_A)
    eigenvalues_A = find_eigenvalues(A_final)
    
    res_sum = check_eigenvalues_sum(A, eigenvalues_A)
    res_prod = check_eigenvalues_product(A, eigenvalues_A)

    B, eps_B = read_qr_system_from_files(MATRIX_B_PATH, EPSILON_PATH)
    B_final, iteration_B = qr_complex_algorithm(B, eps_B)
    eigenvalues_B = extract_all_eigenvalues(B_final)

    sorted_eigval_mine = sort_complex(eigenvalues_B)
    sorted_eigval_numpy = sort_complex(np.linalg.eigvals(B))
    res_B = np.allclose(sorted_eigval_mine, sorted_eigval_numpy)

    report = make_report(A, eps_A, Q, R, A_final, iterations, eigenvalues_A,
                         res_QR, res_orth, res_tri, res_sum, res_prod,
                         B, B_final, iteration_B, eigenvalues_B, res_B)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)
    
    print("Отчет записан в файл:", REPORT_PATH)


if __name__ == "__main__":
    main()
