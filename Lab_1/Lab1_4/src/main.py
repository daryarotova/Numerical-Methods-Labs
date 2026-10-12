import numpy as np
from checks import check_AV_VL, check_orthogonality
from jacobi_rotation import jacobi_rotation_solve
from utils import read_jacobi_system_from_files
from report_utils import add_section, add_matrix, add_value, add_list

MATRIX_A_PATH = "data/matrix_A.txt"
EPSILON_PATH = "data/epsilon.txt"
REPORT_PATH = "data/report_lab_1_4.txt"


def make_report(A, eps, eigenvalues, V, iteration, res_AV_VL, res_ortho_V, eps_table):
    lines = []
    lines.append("Лабораторная работа 1.4: метод вращений Якоби")
    lines.append("")
    add_value(lines, "Размерность системы", A.shape[0])

    add_section(lines, "Входные данные")
    add_matrix(lines, "Матрица A", A)
    add_value(lines, "Точность epsilon", eps)

    add_section(lines, "Выходные данные")
    add_value(lines, "Число итераций", iteration)
    add_value(lines, "Собственные значения", eigenvalues)
    add_matrix(lines, "Матрица собственных векторов V", V)

    add_section(lines, "Проверки")
    add_value(lines, "Проверка A @ V = V @ Λ", res_AV_VL)
    add_value(lines, "Проверка ортогональности матрицы собственных векторов V", res_ortho_V)

    add_section(lines, "Зависимость числа итераций от точности")
    eps_items = [f"epsilon = {eps_i} -> iterations = {iter_i}" for eps_i, iter_i in eps_table]
    add_list(lines, "Результаты", eps_items)

    return "\n".join(lines)


def main():
    A, eps = read_jacobi_system_from_files(MATRIX_A_PATH, EPSILON_PATH)
    eigenvalues, V, iteration = jacobi_rotation_solve(A, eps)

    res_AV_VL = check_AV_VL(A, V, eigenvalues)
    res_ortho_V = check_orthogonality(V)

    eps_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    eps_table = []
    for eps_i in eps_values:
        _, _, iter_i = jacobi_rotation_solve(A, eps_i)
        eps_table.append((eps_i, iter_i))

    report = make_report(A, eps, eigenvalues, V, iteration, res_AV_VL, res_ortho_V, eps_table)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)
    
    print("Отчет записан в файл:", REPORT_PATH)


if __name__ == "__main__":
    main()