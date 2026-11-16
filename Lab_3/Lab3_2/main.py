import matplotlib.pyplot as plt
from spline import (
    read_data, count_steps, find_c_coefficients,
    build_spline_coeffs, evaluate_spline
)


DATA_PATH = "data_3_2.txt"
REPORT_PATH = "report_lab_3_2.txt"
PLOT_PATH = "plot_lab_3_2.png"


def make_report(x_nodes, y_nodes, coeffs, x_given):
    lines = [
            "Лабораторная работа 3.2: Кубический сплайн",
            "Вариант 18\n"
            f"Точка интерполяции: x* = {x_given}",
        ]

    lines.append("\n1. Исходные узлы и значения функции:")
    for i in range(len(x_nodes)):
        lines.append(f"   x_{i} = {x_nodes[i]:.4f},  y_{i} = {y_nodes[i]:.4f}")

    lines.append("\n2, 3. Коэффициенты сплайна и интервалы их действия:")
    lines.append("  i |   интервал   |     a     |     b     |     c     |     d     ")
    for i in range(len(coeffs)):
        a, b, c, d = coeffs[i]
        lines.append(f"  {i+1} | [{x_nodes[i]:4.2f}, {x_nodes[i+1]:4.2f}] | {a:9.5f} | {b:9.5f} | {c:9.5f} | {d:9.5f}")

    s_x_given, interval_idx = evaluate_spline(x_given, x_nodes, coeffs)
    lines.append(f"\n4. Значение сплайна в точке x* = {x_given}:")
    lines.append(f"   S(x*) = {s_x_given:.8f}")
    lines.append(f"\n5. Номер интервала, содержащего x*: {interval_idx}")

    lines.append("\n6. Проверка прохождения сплайна через исходные узлы:")
    lines.append("   i | x_i | S(x_i)  | y(x_i)  | S(x_i) - y(x_i) ")
    for i in range(len(x_nodes)):
        s_val, _ = evaluate_spline(x_nodes[i], x_nodes, coeffs)
        diff = abs(s_val - y_nodes[i])
        lines.append(f"   {i+1} | {x_nodes[i]} | {s_val:.5f} | {y_nodes[i]:.5f} | {diff:.4f}")

    c_left = coeffs[0][2]
    s2_left = 2 * c_left

    c_right = coeffs[-1][2]
    d_right = coeffs[-1][3]
    h_right = x_nodes[-1] - x_nodes[-2]
    s2_right = 2 * c_right + 6 * d_right * h_right

    lines.append("\n7. Значения второй производной на концах интервала:")
    lines.append(f"   Вторая производная левой границы  - S''(x_0) = {s2_left:.5e}")
    lines.append(f"   Вторая производная правой границы - S''(x_4) = {s2_right:.5e}")

    return "\n".join(lines)


def plot_spline(x_nodes, y_nodes, coeffs, x_given):
    n_points = 200
    x_min, x_max = x_nodes[0], x_nodes[-1]
    step = (x_max - x_min) / (n_points - 1)
    grid_x = [x_min + i * step for i in range(n_points)]
    
    grid_y = [evaluate_spline(x, x_nodes, coeffs)[0] for x in grid_x]
    s_star, _ = evaluate_spline(x_given, x_nodes, coeffs)

    plt.figure(figsize=(10, 6))
    plt.plot(grid_x, grid_y, 'k-', linewidth=2, label='Кубический сплайн S(x)')
    plt.plot(x_nodes, y_nodes, 'o', color='m', markersize=6, label='Исходные узлы')
    plt.plot(x_given, s_star, 'o', color='orange', markersize=8, label=f'x* = {x_given}, S(x*) = {s_star:.4f}')
    plt.axvline(x=x_given, color='gray', linestyle='--', alpha=0.5)

    x_ticks = [i * 0.5 for i in range(int(x_max * 2) + 2)]
    plt.xticks(x_ticks)

    y_max = max(grid_y)
    y_ticks = [i * 1.0 for i in range(int(y_max) + 2)]
    plt.yticks(y_ticks)

    plt.title('Кубический сплайн')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main():
    x_nodes, y_nodes, x_given = read_data(DATA_PATH)
    h = count_steps(x_nodes)
    c_full = find_c_coefficients(h, y_nodes)
    coeffs = build_spline_coeffs(x_nodes, y_nodes, h, c_full)

    report = make_report(x_nodes, y_nodes, coeffs, x_given)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nОтчёт записан в файл: {REPORT_PATH}")
    
    plot_spline(x_nodes, y_nodes, coeffs, x_given)
    print(f"\nГрафик сохранён в файл: {PLOT_PATH}\n")


if __name__ == "__main__":
    main()