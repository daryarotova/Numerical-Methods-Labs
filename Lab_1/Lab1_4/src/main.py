import numpy as np
from checks import check_AV_VL, check_orthogonality
from jacobi_rotation import jacobi_rotation_solve
from utils import read_jacobi_system_from_files


REPORT_PATH = "data/report_lab_1_4.txt"


def make_report(A, eps, eigenvalues, V, iteration, res_AV_VL, res_ortho_V, eps_table):
    lines = []
    lines.append("Лабораторная работа 1.4: метод вращений Якоби")
    lines.append("")
    lines.append(f"Размерность системы: n = {A.shape[0]}")
    lines.append("")

    lines.append("--- Входные данные: ---")
    lines.append("Матрица A =")
    lines.append(str(A))
    lines.append("")

    lines.append(f"Точность epsilon = {eps}")
    lines.append("")

    lines.append("--- Выходные данные: ---")
    lines.append(f"Число итераций: {iteration}")
    
    lines.append("Собственные значения =")
    lines.append(str(eigenvalues))
    lines.append("")

    lines.append("Матрица собственных векторов V =")
    lines.append(str(V))
    lines.append("")

    lines.append("--- Проверки: ---")
    lines.append(f"Проверка A · V = V · Λ: {res_AV_VL}")
    lines.append(f"Проверка ортогональности матрицы С.В. V: {res_ortho_V}")
    lines.append("")

    lines.append("--- Зависимость числа итераций от точности: ---")
    for eps_i, iter_i in eps_table:
        lines.append(f"epsilon = {eps_i} - iterations = {iter_i}")

    return "\n".join(lines)


def main():
    A, eps = read_jacobi_system_from_files("data/matrix_A.txt", "data/epsilon.txt")
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