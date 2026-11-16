from numerical_differentiation import (
    read_data, 
    table_of_divided_differences, 
    calculate_derivatives_newton, 
    calculate_derivatives_local
)


DATA_PATH = "data_3_4.txt"
REPORT_PATH = "report_lab_3_4.txt"


def make_report(x_nodes, y_nodes, x_given, n_d1, n_d2, l_d1_left, l_d1_right, l_d1_central, l_d2):
    lines = [
        "Лабораторная работа 3.4: Численное дифференцирование",
        "Вариант 18\n"
    ]
    lines.append("1. Исходная таблица значений функции:")
    lines.append("  i |   x_i   |   y_i   ")
    for i in range(len(x_nodes)):
        lines.append(f"  {i} | {x_nodes[i]:7.4f} | {y_nodes[i]:7.4f}")
    lines.append("")
    
    lines.append(f"2. Точка, в которой выполняется вычисление: \n  x* = {x_given}\n")

    lines.append("3. Вычисление через многочлен Ньютона:")
    lines.append(" Использованы все 5 узлов: x = [-0.2, 0.0, 0.2, 0.4, 0.6]")
    lines.append(f" Первая производная:  f'(x*) = {n_d1:.6f}")
    lines.append(f" Вторая производная:  f''(x*) = {n_d2:.6f}\n")

    lines.append("4. Вычисление через локальные конечные разности:")
    lines.append(" Использованы 3 центральных узла: x = [0.0, 0.2, 0.4]")
    lines.append(f" Левосторонняя:       f'(x*) = {l_d1_left:.6f}")
    lines.append(f" Правосторонняя:      f'(x*) = {l_d1_right:.6f}")
    lines.append(f" Центральная:         f'(x*) = {l_d1_central:.6f}")
    lines.append(f" Вторая производная:  f''(x*) = {l_d2:.6f}\n")

    return "\n".join(lines)

def main():
    x_nodes, y_nodes, x_given = read_data(DATA_PATH)
    table = table_of_divided_differences(x_nodes, y_nodes)

    n_d1, n_d2 = calculate_derivatives_newton(x_nodes, table, x_given)
    l_d1_left, l_d1_right, l_d1_central, l_d2 = calculate_derivatives_local(x_nodes, y_nodes, x_given)

    report = make_report(
        x_nodes, y_nodes, x_given, 
        n_d1, n_d2, 
        l_d1_left, l_d1_right, l_d1_central, l_d2
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nОтчёт записан в файл: {REPORT_PATH}\n")


if __name__ == "__main__":
    main()