import math

EPSILON_PATH = "data/epsilon.txt"
REPORT_PATH = "data/report_lab_2_1.txt"


def read_epsilon(path):
    with open(path, "r", encoding="utf-8") as file:
        return float(file.read().strip())
    

def f(x):
    """Исходная функция f(x) = ln(x+1) - x^3 + 1"""
    return math.log(x + 1) - x**3 + 1


def df(x):
    """Первая производная f'(x) = 1/(x+1) - 3x^2"""
    return 1.0 / (x + 1) - 3 * x**2


def ddf(x):
    """Вторая производная f''(x) = -1/(x+1)^2 - 6x"""
    return -1.0 / ((x + 1)**2) - 6 * x


def phi_lam(x, lam):
    """Функция phi(x) = x - lambda * f(x)"""
    return x - lam * f(x)


def dphi_lam(x, lam):
    """Производная phi'(x) = 1 - lambda * f'(x)"""
    return 1.0 - lam * df(x)


def simple_iteration_method(x0, a, b, eps, max_iter=1000):
    lam = 1.0 / df(x0)

    q = max(abs(dphi_lam(a, lam)), abs(dphi_lam(b, lam)))
    if q >= 1:
        raise ValueError(f"Условие сходимости не выполнено: q = {q} >= 1, а должно быть < 1")

    x_current = x0
    steps = []

    for k in range(1, max_iter + 1):
        x_next = phi_lam(x_current, lam)
        error_rate = abs(x_next - x_current)
        steps.append((k, x_next, f(x_next), error_rate))
        eps_k = (q / (1 - q)) * error_rate
        
        if eps_k <= eps:
            return x_next, k, steps, q
            
        x_current = x_next

    raise RuntimeError(f"Метод простой итерации не сошелся за {max_iter} итераций")


def newton_method(x0, eps, max_iter=1000):
    x_current = x0
    steps = []

    for k in range(1, max_iter + 1):
        fx = f(x_current)
        dfx = df(x_current)

        if abs(dfx) < 1e-12:
            raise ValueError(f"Производная близка к нулю на итерации {k}")

        x_next = x_current - (fx / dfx)
        error_rate = abs(x_next - x_current)        
        steps.append((k, x_next, f(x_next), error_rate))
        
        if error_rate < eps:
            return x_next, k, steps
        
        x_current = x_next
        
    raise RuntimeError(f"Метод Ньютона не сошелся за {max_iter} итераций")


def make_report(x0_s, root_s, k_s, steps_s, q, a, b, x0_n, root_n, k_n, steps_n, eps,
                 f_12, ddf_12, check_12, f_13, ddf_13, check_13):
    lines = []
    lines.append("Лабораторная работа 2.1: Решение нелинейного уравнения\n")
    lines.append("Вариант 18: ln(x+1) - x^3 + 1 = 0")
    lines.append(f"Точность epsilon = {eps}\n")

    lines.append("--- Проверка начальных приближений ---")

    lines.append("          Метод простой итерации\nПо теореме 2.3: q < 1:")
    lines.append(f"Корень x принадлежит отрезку [{a}, {b}].")
    lines.append(f"На этом отрезке коэффициент сходимости q = {q:.4f} < 1.")
    lines.append(f"Вывод: x0 = {x0_s}.\n")
    
    lines.append("          Метод Ньютона\nПо теореме 2.2: f(x0) * f''(x0) > 0:")
    lines.append(f"При x0 = {x0_s}: f({x0_s}) * f''({x0_s}) = {f_12:.4f} * {ddf_12:.4f} = {check_12:.4f} < 0 - не подходит")
    lines.append(f"При x0 = {x0_n}: f({x0_n}) * f''({x0_n}) = {f_13:.4f} * {ddf_13:.4f} = {check_13:.4f} > 0 - подходит")
    lines.append(f"Вывод: x0 = {x0_n}.\n")

    lines.append("--- Метод простой итерации ---")
    lines.append(f"Начальное приближение: x0 = {x0_s}")
    lines.append(f"Коэффициент: q = {q:.4f}\n")
    lines.append("k\t x_k\t\t f(x_k)\t\t Погрешность")
    for k, x, fx, delta in steps_s:
        lines.append(f"{k}\t {x:.8f}\t {fx:.2e}\t {delta:.2e}")
    lines.append(f"\nКорень: {root_s:.8f}")
    lines.append(f"Количество итераций: {k_s}\n")

    lines.append("--- Метод Ньютона ---")
    lines.append(f"Начальное приближение: x0 = {x0_n}\n")
    lines.append("k\t x_k\t\t f(x_k)\t\t Погрешность")
    for k, x, fx, delta in steps_n:
        lines.append(f"{k}\t {x:.8f}\t {fx:.2e}\t {delta:.2e}")
    lines.append(f"\nКорень: {root_n:.8f}")
    lines.append(f"Количество итераций: {k_n}\n")

    lines.append("--- Сравнение методов ---")
    lines.append("Метод Ньютона демонстрирует более быструю сходимость, \n"
    "так как его погрешность убывает быстрее, чем у метода простой итерации.")
    
    return "\n".join(lines)


def main():
    eps = read_epsilon(EPSILON_PATH)
    x0_simple = 1.2
    x0_newton = 1.3

    f_12 = f(1.2)
    ddf_12 = ddf(1.2)
    check_12 = f_12 * ddf_12
    
    f_13 = f(1.3)
    ddf_13 = ddf(1.3)
    check_13 = f_13 * ddf_13

    a, b = 1.2, 1.3

    root_s, k_s, steps_s, q = simple_iteration_method(x0_simple, a, b, eps)
    root_n, k_n, steps_n = newton_method(x0_newton, eps)

    report = make_report(x0_simple, root_s, k_s, steps_s, q, a, b,
                         x0_newton, root_n, k_n, steps_n, eps,
                         f_12, ddf_12, check_12, f_13, ddf_13, check_13)
     
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)
        
    print(f"Отчет записан в файл: {REPORT_PATH}")


if __name__ == "__main__":
    main()