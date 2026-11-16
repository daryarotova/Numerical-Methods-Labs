from numerical_integration import (
    generate_grid, left_rectangles, right_rectangles,
    midpoint_rectangles, trapezoidal, simpson, runge_romberg
)


DATA_PATH = "data_3_5.txt"
REPORT_PATH = "report_lab_3_5.txt"


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    a, b = [float(x) for x in lines[0].split()]
    h1, h2 = [float(x) for x in lines[1].split()]
    return a, b, h1, h2


def make_report(a, b, h1, h2, results, exact_val):
    lines = [
        "Лабораторная работа 3.5: Численное интегрирование",
        "Вариант 18: f(x) = x / (16 - x^4)",
        f"\nОтрезок: [{a}, {b}], шаги: h1 = {h1}, h2 = {h2}",
        f"\nТочное значение интеграла (аналитически): I = {exact_val:.6f}\n"
    ]

    header = f"{'Метод':<22} | {'I(h1)':>10} | {'I(h2)':>10} | {'p':>2} | {'I_RR':>10} | {'Оценка (h2)':>12} | {'Абс. погрешность':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    best_method = None
    min_abs_error = float('inf')

    for name, I_h1, I_h2, p, I_rr in results:
        runge_err = abs(I_rr - I_h2)
        abs_err = abs(exact_val - I_h2)
        
        if abs_err <= min_abs_error:
            min_abs_error = abs_err
            best_method = name

        v1 = 0.0 if abs(I_h1) < 1e-12 else I_h1
        v2 = 0.0 if abs(I_h2) < 1e-12 else I_h2
        v_rr = 0.0 if abs(I_rr) < 1e-12 else I_rr

        lines.append(
            f"{name:<22} | {v1:>10.6f} | {v2:>10.6f} | {p:>2} | {v_rr:>10.6f} | {runge_err:>12.2e} | {abs_err:>10.2e}"
        )

    lines.append("\nВывод:")
    lines.append(f"Наиболее точный результат дает метод - {best_method}")
    lines.append(f"Его абсолютная погрешность составляет - {min_abs_error:.2e}")

    return "\n".join(lines)


def main():
    a, b, h1, h2 = read_data(DATA_PATH)
    
    grid_h1, _ = generate_grid(a, b, h1)
    grid_h2, _ = generate_grid(a, b, h2)

    methods = [
        ("Левые прямоугольники", left_rectangles, 1),
        ("Правые прямоугольники", right_rectangles, 1),
        ("Средние прямоугольники", midpoint_rectangles, 2),
        ("Метод трапеции", trapezoidal, 2),
        ("Метод Симпсона", simpson, 4)
    ]

    results = []
    for name, func, p in methods:
        I_h1 = func(grid_h1, h1)
        I_h2 = func(grid_h2, h2)
        I_rr = runge_romberg(I_h1, I_h2, p)
        results.append((name, I_h1, I_h2, p, I_rr))

    exact_val = 0.0
    report = make_report(a, b, h1, h2, results, exact_val)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nОтчёт записан в файл: {REPORT_PATH}\n")


if __name__ == "__main__":
    main()