import matplotlib.pyplot as plt
from least_squares import (
    read_data, build_system, format_polynomial, calculate_mse, evaluate
)


DATA_PATH = "data_3_3.txt"
REPORT_PATH = "report_lab_3_3.txt"
PLOT_PATH = "plot_lab_3_3.png"


def make_report(results):
    lines = [
        "Лабораторная работа 3.3: Метод наименьших квадратов",
        "Вариант 18\n"
    ]

    for deg, coeffs, mse, poly_str in results:
        lines.append(f"{deg}. Многочлен {deg}-й степени:")
        lines.append(f"   Коэффициенты: {[round(c, 6) for c in coeffs]}")
        lines.append(f"   Явный вид:    P_{deg}(x) = {poly_str}")
        lines.append(f"   Значение MSE_{deg} = {mse:.4e}\n")

    lines.append("Сравнение ошибок:")
    for deg, _, mse, _ in results:
        lines.append(f"   MSE_{deg} = {mse:.4e}")

    best_deg, _, best_mse, _ = min(results, key=lambda x: x[2])
    lines.append(f"\nВывод: многочлен {best_deg}-й степени лучше приближает исходные данные,")
    lines.append(f"так как у него наименьшая ошибка MSE = {best_mse:.4e}.")

    return "\n".join(lines)


def plot_least_squares(x_nodes, y_nodes, results):
    n_points = 200
    x_min, x_max = min(x_nodes), max(x_nodes)
    step = (x_max - x_min) / (n_points - 1)
    grid_x = [x_min + i * step for i in range(n_points)]
    grid_y1 = [evaluate(results[0][1], x) for x in grid_x]
    grid_y2 = [evaluate(results[1][1], x) for x in grid_x]
    grid_y3 = [evaluate(results[2][1], x) for x in grid_x]

    plt.figure(figsize=(10, 6))
    plt.plot(x_nodes, y_nodes, 'ko', markersize=7, label='Исходные точки', zorder=5)
    
    plt.plot(grid_x, grid_y1, "r--", linewidth=2, label="Многочлен 1-й степени")
    plt.plot(grid_x, grid_y2, "g-.", linewidth=2, label="Многочлен 2-й степени")
    plt.plot(grid_x, grid_y3, "b:", linewidth=2, label="Многочлен 3-й степени")

    plt.title('Метод наименьших квадратов')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main():
    x_nodes, y_nodes = read_data(DATA_PATH)
    degrees = [1, 2, 3]

    results = []
    for deg in degrees:
        coeffs = build_system(x_nodes, y_nodes, deg)
        mse = calculate_mse(x_nodes, y_nodes, coeffs)
        poly_str = format_polynomial(coeffs)
        results.append((deg, coeffs, mse, poly_str))

    report = make_report(results)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nОтчёт записан в файл: {REPORT_PATH}")
    
    plot_least_squares(x_nodes, y_nodes, results)
    print(f"\nГрафик сохранён в файл: {PLOT_PATH}\n")


if __name__ == "__main__":
    main()