import numpy as np
from checks import check_stability, check_sweep_solution
from sweep import sweep_backward, sweep_forward
from utils import read_sweep_system_from_files


REPORT_PATH = "data/report_lab_1_2.txt"


def make_report(a, b, c, d, P, Q, x, res_stability, res_solution):
    lines = []
    lines.append("Лабораторная работа 1.2: Метод прогонки")
    lines.append("")
    lines.append(f"Размерность системы: n = {b.shape[0]}")
    lines.append("")

    lines.append("--- Входные данные: ---")
    lines.append("Вектор a =")
    lines.append(str(a))
    lines.append("")

    lines.append("Вектор b =")
    lines.append(str(b))
    lines.append("")

    lines.append("Вектор c =")
    lines.append(str(c))
    lines.append("")

    lines.append("Вектор d =")
    lines.append(str(d))
    lines.append("")

    lines.append("--- Выходные данные: ---")
    lines.append("Прогоночные коэффициенты P =")
    lines.append(str(P))
    lines.append("")

    lines.append("Прогоночные коэффициенты Q =")
    lines.append(str(Q))
    lines.append("")

    lines.append("Решение системы x =")
    lines.append(str(x))
    lines.append("")

    lines.append("--- Проверки: ---")
    lines.append(f"Диагональное преобладание: {res_stability}")
    lines.append("")
    lines.append(f"Невязка системы: {res_solution}")

    return "\n".join(lines)


def main():
    a, b, c, d = read_sweep_system_from_files("data/vector_a.txt", "data/vector_b.txt", 
                                              "data/vector_c.txt", "data/vector_d.txt")
    
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