import math


EPSILON_PATH = "data/epsilon.txt"
REPORT_PATH = "data/report_lab_2_2.txt"


def read_epsilon(path):
    with open(path, "r", encoding="utf-8") as file:
        return float(file.read().strip())


def f1(x1, x2):
    return 4 * x1 - math.cos(x2)


def f2(x1, x2):
    return 4 * x2 - math.exp(x1)


def jacobian(x1, x2):
    j11 = 4.0
    j12 = math.sin(x2)
    j21 = -math.exp(x1)
    j22 = 4.0
    return j11, j12, j21, j22


def phi1(x1, x2):
    return math.cos(x2) / 4.0


def phi2(x1, x2):
    return math.exp(x1) / 4.0


def dphi1(x1, x2):
    return -math.sin(x2) / 4.0


def dphi2(x1, x2):
    return math.exp(x1) / 4.0


def newton_method_system(x1_0, x2_0, eps, max_iter=100):
    x1_curr, x2_curr = x1_0, x2_0
    steps = []

    for k in range(1, max_iter + 1):
        vec_f1 = f1(x1_curr, x2_curr)
        vec_f2 = f2(x1_curr, x2_curr)

        j11, j12, j21, j22 = jacobian(x1_curr, x2_curr)

        detJ = j11 * j22 - j12 * j21
        if abs(detJ) < 1e-12:
            raise ValueError(f"Определитель матрицы Якоби близок к нулю.")        
        
        det_x1 = -vec_f1 * j22 + j12 * vec_f2
        det_x2 = -vec_f2 * j11 + j21 * vec_f1
        delta_x1 = det_x1 / detJ
        delta_x2 = det_x2 / detJ

        x1_next = x1_curr + delta_x1
        x2_next = x2_curr + delta_x2

        error_rate = max(abs(delta_x1), abs(delta_x2))
        steps.append((k, x1_next, x2_next, error_rate))

        if error_rate < eps:
            return x1_next, x2_next, k, steps

        x1_curr, x2_curr = x1_next, x2_next

    raise RuntimeError(f"Метод Ньютона для системы не сошелся за {max_iter} итераций")


def simple_iteration_method_system(x1_0, x2_0, eps, max_iter=1000):
    q = max(abs(dphi1(x1_0, x2_0)), abs(dphi2(x1_0, x2_0)))
    if q >= 1:
        raise ValueError(f"Условие сходимости не выполнено: q = {q} >= 1, а должно быть < 1")
    
    x1_curr, x2_curr = x1_0, x2_0
    steps = []

    for k in range(1, max_iter + 1):
        x1_next = phi1(x1_curr, x2_curr)
        x2_next = phi2(x1_curr, x2_curr)

        error_rate = max(abs(x1_next - x1_curr), abs(x2_next - x2_curr))
        steps.append((k, x1_next, x2_next, error_rate))

        eps_k = q / (1 - q) * error_rate
        if eps_k <= eps:
            return x1_next, x2_next, k, steps, q
        
        x1_curr, x2_curr = x1_next, x2_next

    raise RuntimeError(f"Метод простой итерации для системы не сошелся за {max_iter} итераций")


def make_report(x1_n, x2_n, k_n, steps_n, x1_s, x2_s, k_s, steps_s, q, eps, x1_0, x2_0):
    lines = []
    lines.append("Лабораторная работа 2.2: Решение системы нелинейных уравнений\n")
    lines.append("Вариант 18: \n  4x1 - cos(x2) = 0;\n  4x2 - exp(x1) = 0")
    lines.append(f"Точность epsilon = {eps}\n")

    lines.append("--- Метод простой итерации ---")
    lines.append(f"Начальное приближение: x1 = {x1_0}, x2 = {x2_0}")
    lines.append(f"Коэффициент q = {q:.4f}\n")
    lines.append("k\t\t\t x1\t\t\t\t\t x2\t\t\t Погрешность")
    for k, x1, x2, delta in steps_s:
        lines.append(f"{k}\t {x1:.8f}\t {x2:.8f}\t {delta:.2e}")
    lines.append(f"\nРешение: x1 = {x1_s:.8f}, x2 = {x2_s:.8f}")
    lines.append(f"Количество итераций: {k_s}\n")

    lines.append("--- Метод Ньютона ---")
    lines.append(f"Начальное приближение: x1 = {x1_0}, x2 = {x2_0}\n")
    lines.append("k\t\t\t x1\t\t\t\t\t x2\t\t\t Погрешность")
    for k, x1, x2, delta in steps_n:
        lines.append(f"{k}\t {x1:.8f}\t {x2:.8f}\t {delta:.2e}")
    lines.append(f"\nРешение: x1 = {x1_n:.8f}, x2 = {x2_n:.8f}")
    lines.append(f"Количество итераций: {k_n}\n")
    
    lines.append("--- Сравнение методов ---")
    err_n = steps_n[-1][3]
    err_s = steps_s[-1][3]
    lines.append(f"Метод Ньютона: погрешность снизилась до {err_n:.2e} за {k_n} итераций.")
    lines.append(f"Метод простой итерации: погрешность снизилась до {err_s:.2e} за {k_s} итераций.")
    lines.append("Вывод: метод Ньютона сходится быстрее.")
    
    return "\n".join(lines)


def main():
    eps = read_epsilon(EPSILON_PATH)
    
    x1_0 = 0.24
    x2_0 = 0.31
    
    x1_s, x2_s, k_s, steps_s, q = simple_iteration_method_system(x1_0, x2_0, eps)
    x1_n, x2_n, k_n, steps_n = newton_method_system(x1_0, x2_0, eps)
    
    report = make_report(x1_n, x2_n, k_n, steps_n, x1_s, x2_s, k_s, steps_s, q, eps, x1_0, x2_0)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)
        
    print(f"Отчет записан в файл: {REPORT_PATH}")


if __name__ == "__main__":
    main()