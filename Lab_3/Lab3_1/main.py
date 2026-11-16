import matplotlib.pyplot as plt
from interpolation import (
    read_data, function, lagrange_evaluate, 
    table_of_divided_differences, newton_evaluate,
    lagrange_polynomial_string, newton_polynomial_string
)


DATA_PATH = "data_3_1.txt"
REPORT_PATH = "report_lab_3_1.txt"
PLOT_PATH = "plot_lab_3_1.png"


def format_divided_table(x_nodes, table):
    n = len(x_nodes)
    header = " i | x_i    | " + " | ".join(f"{k}-й пор." for k in range(n))
    lines = [header, "-" * len(header)]
    for i in range(n):
        row = f"{i:>2} | {x_nodes[i]:.4f}"
        for k in range(n - i):
            row += f" | {table[i][k]:.6f}"
        lines.append(row)
    return "\n".join(lines)


def make_variant_report(title, x_nodes, y_nodes, x_given, y_exact):
    tbl = table_of_divided_differences(x_nodes, y_nodes)
    L = lagrange_evaluate(x_nodes, y_nodes, x_given)
    P = newton_evaluate(x_nodes, tbl, x_given)
    n = len(x_nodes)

    lines = [f"\nВариант {title}\n"]

    lines.append("1. Узлы интерполяции и значения функции:")
    lines.append("   i | x_i    | y_i")
    for i in range(n):
        lines.append(f"   {i} | {x_nodes[i]:.4f} | {y_nodes[i]:.8f}")

    lines.append("\n2. Интерполяционный многочлен Лагранжа:")
    lines.append("  " + lagrange_polynomial_string(x_nodes, y_nodes))

    lines.append("\n3. Таблица разделённых разностей:")
    lines.append(format_divided_table(x_nodes, tbl))

    lines.append("\n4. Интерполяционный многочлен Ньютона:")
    lines.append("  " + newton_polynomial_string(x_nodes, tbl))

    lines.append(f"\n5. Результаты в точке x* = {x_given}:")
    lines.append(f"  Многочлен Лагранжа:        L(x*) = {L:.8f}")
    lines.append(f"  Многочлен Ньютона:         P(x*) = {P:.8f}")
    lines.append(f"  Точное значение функции:   y(x*) = {y_exact:.8f}")

    lines.append("\n6. Абсолютные погрешности:")
    lines.append(f"  |y(x*) - L(x*)| = {abs(y_exact - L):.4e}")
    lines.append(f"  |y(x*) - P(x*)| = {abs(y_exact - P):.4e}")

    lines.append("\n7. Разность между результатами методов:")
    lines.append(f"   |L(x*) - P(x*)| = {abs(L - P):.4e}")

    return "\n".join(lines), L, P


def make_report(x_nodes_a, y_nodes_a, x_nodes_b, y_nodes_b, x_given):
    y_exact = function(x_given)

    lines = [
            "Лабораторная работа 3.1: Интерполяционные многочлены Лагранжа и Ньютона",
            "Вариант 18: y(x) = sqrt(x) + x",
            f"Точка интерполяции: x* = {x_given}",
            f"Точное значение: y(x*) = {y_exact:.10f}",
        ]

    report_a, L_a, _ = make_variant_report("а - Равномерные узлы", x_nodes_a, y_nodes_a, x_given, y_exact)
    report_b, L_b, _ = make_variant_report("б - Неравномерные узлы", x_nodes_b, y_nodes_b, x_given, y_exact)

    lines.append(report_a)
    lines.append(report_b)

    lines.append("\nСравнение вариантов:")
    lines.append(f"   А: L(x*) = {L_a:.8f}, |y(x*) - L(x*)| = {abs(y_exact - L_a):.4e}")
    lines.append(f"   Б: L(x*) = {L_b:.8f}, |y(x*) - L(x*)| = {abs(y_exact - L_b):.4e}")

    if abs(y_exact - L_a) < abs(y_exact - L_b):
        lines.append("\nВывод: равномерные узлы дают меньшую погрешность.")
    elif abs(y_exact - L_b) < abs(y_exact - L_a):
        lines.append("\nВывод: неравномерные узлы дают меньшую погрешность.")
    else:
        lines.append("\nВывод: оба варианта дают одинаковую погрешность.")
    return "\n".join(lines)


def plot_interpolation(x_nodes_a, y_nodes_a, x_nodes_b, y_nodes_b, x_given):
    n_points = 200
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xs, ys, title in [
        (axes[0], x_nodes_a, y_nodes_a, "а - Равномерные узлы"),
        (axes[1], x_nodes_b, y_nodes_b, "б - Неравномерные узлы"),
    ]:
        x_min, x_max = xs[0], xs[-1]
        step = (x_max - x_min) / (n_points - 1)
        grid_x = [x_min + i * step for i in range(n_points)]

        tbl = table_of_divided_differences(xs, ys)
        grid_f = [function(x) for x in grid_x]
        grid_L = [lagrange_evaluate(xs, ys, x) for x in grid_x]
        grid_P = [newton_evaluate(xs, tbl, x) for x in grid_x]

        ax.plot(grid_x, grid_f, "k-", linewidth=2, label="f(x) = sqrt(x) + x", alpha=0.6)
        ax.plot(grid_x, grid_L, "c--", label="L(x) - Многочлен Лагранжа")
        ax.plot(grid_x, grid_P, "m:", label="P(x) - Многочлен Ньютона")
        ax.plot(xs, ys, "o", color="olive", markersize=6, label="Узлы интерполяции", zorder=5)
        ax.plot(x_given, function(x_given), "o", color="sandybrown", markersize=8, label=f"x* = {x_given}", zorder=6)
        ax.axvline(x=x_given, color="gray", linestyle="--", alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main():
    x_nodes_a, x_nodes_b, x_given = read_data(DATA_PATH)
    y_nodes_a = [function(x) for x in x_nodes_a]
    y_nodes_b = [function(x) for x in x_nodes_b]

    report = make_report(x_nodes_a, y_nodes_a, x_nodes_b, y_nodes_b, x_given)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nОтчёт записан в файл: {REPORT_PATH}")
    plot_interpolation(x_nodes_a, y_nodes_a, x_nodes_b, y_nodes_b, x_given)
    print(f"\nГрафик сохранён в файл: {PLOT_PATH}")


if __name__ == "__main__":
    main()
