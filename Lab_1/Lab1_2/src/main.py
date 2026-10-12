import numpy as np
from checks import check_stability, check_sweep_solution
from sweep import sweep_backward, sweep_forward
from utils import read_sweep_system_from_files
from report_utils import add_section, add_matrix, add_value, add_list


REPORT_PATH = "data/report_lab_1_2.txt"
FILE_A = "data/vector_a.txt"
FILE_B = "data/vector_b.txt"
FILE_C = "data/vector_c.txt"
FILE_D = "data/vector_d.txt"


def make_report(a, b, c, d, P, Q, x, res_stability, res_solution):
    lines = []
    lines.append("Лабораторная работа 1.2: Метод прогонки")
    lines.append("")
    add_value(lines, "Размерность системы", b.shape[0])

    add_section(lines, "Входные данные")
    add_matrix(lines, "Вектор a", a)
    add_matrix(lines, "Вектор b", b)
    add_matrix(lines, "Вектор c", c)
    add_matrix(lines, "Вектор d", d)

    add_section(lines, "Выходные данные")
    add_matrix(lines, "Прогоночные коэффициенты P", P)
    add_matrix(lines, "Прогоночные коэффициенты Q", Q)
    add_matrix(lines, "Решение системы x", x)

    add_section(lines, "Проверки")
    add_value(lines, "Диагональное преобладание", res_stability)
    add_value(lines, "Невязка системы", res_solution)

    return "\n".join(lines)


def main():
    a, b, c, d = read_sweep_system_from_files(FILE_A, FILE_B, FILE_C, FILE_D)
    
    P, Q = sweep_forward(a, b, c, d)
    x = sweep_backward(P, Q)

    res_stability = check_stability(a, b, c)
    res_solution = check_sweep_solution(a, b, c, d, x)

    report = make_report(a, b, c, d, P, Q, x, res_stability, res_solution)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print("Отчет записан в файл:", REPORT_PATH)


if __name__ == "__main__":
    main()