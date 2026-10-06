import numpy as np
from checks import check_convergence_condition, check_diagonal_dominance, check_solution, solve_reference
from solve import jacobi_solve, seidel_solve, build_equivalent_form
from utils import read_iteration_system_from_files


REPORT_PATH = "data/report_lab_1_3.txt"


def make_report(A, b, eps, alpha, beta, norm_alpha, converged, diag_dominance,
                x_jacobi, k_jacobi, x_seidel, k_seidel, x_ref,
                res_jacobi, res_seidel, err_jacobi, err_seidel):
    lines = []
    lines.append("Лабораторная работа 1.3: метод простых итераций и метод Зейделя")
    lines.append("")
    lines.append(f"Размерность системы: n = {A.shape[0]}")
    lines.append("")

    lines.append("--- Входные данные: ---")
    lines.append("Матрица A =")
    lines.append(str(A))
    lines.append("")

    lines.append("Вектор b =")
    lines.append(str(b))
    lines.append("")

    lines.append(f"Точность epsilon = {eps}")
    lines.append("")

    lines.append("--- Эквивалентная форма: ---")
    lines.append("Матрица alpha =")
    lines.append(str(alpha))
    lines.append("")

    lines.append("Вектор beta =")
    lines.append(str(beta))
    lines.append("")

    lines.append(f"Норма матрицы alpha ||alpha||_c = {norm_alpha}")
    lines.append("")

    lines.append("--- Проверка условий сходимости: ---")
    lines.append(f"Условие сходимости (||alpha||_c < 1): {converged}")
    lines.append(f"Диагональное преобладание матрицы A: {diag_dominance}")
    lines.append("")

    lines.append("--- Метод простых итераций (Якоби): ---")
    lines.append("Решение x =")
    lines.append(str(x_jacobi))
    lines.append("")
    lines.append(f"Число итераций: {k_jacobi}")
    lines.append(f"Невязка ||A @ x - b||_c: {res_jacobi:.3e}")
    lines.append(f"Отклонение от эталона ||x - x_ref||_c: {err_jacobi:.3e}")
    lines.append("")

    lines.append("--- Метод Зейделя: ---")
    lines.append("Решение x =")
    lines.append(str(x_seidel))
    lines.append("")
    lines.append(f"Число итераций: {k_seidel}")
    lines.append(f"Невязка ||A @ x - b||_c: {res_seidel:.3e}")
    lines.append(f"Отклонение от эталона ||x - x_ref||_c: {err_seidel:.3e}")
    lines.append("")

    lines.append("--- Эталонное решение: ---")
    lines.append(str(x_ref))
    lines.append("")

    lines.append("--- Сравнение методов: ---")
    lines.append(f"Число итераций метода Якоби:   {k_jacobi}")
    lines.append(f"Число итераций метода Зейделя: {k_seidel}")
    lines.append(f"Зейдель быстрее Якоби в {k_jacobi / k_seidel:.2f} раза")

    return "\n".join(lines)


def main():
    A, b, epsilon = read_iteration_system_from_files(
        "data/matrix_A.txt", "data/vector_b.txt", "data/epsilon.txt"
    )
    alpha, beta = build_equivalent_form(A, b)

    norm_alpha, converged = check_convergence_condition(alpha)
    diag_dominance = check_diagonal_dominance(A)

    x_jacobi, k_jacobi = jacobi_solve(alpha, beta, epsilon)
    x_seidel, k_seidel = seidel_solve(alpha, beta, epsilon)
    x_ref = solve_reference(A, b)

    res_jacobi = check_solution(A, b, x_jacobi)
    res_seidel = check_solution(A, b, x_seidel)

    err_jacobi = np.max(np.abs(x_jacobi - x_ref))
    err_seidel = np.max(np.abs(x_seidel - x_ref))
    
    report = make_report(A, b, epsilon, alpha, beta, norm_alpha, converged, diag_dominance, 
                         x_jacobi, k_jacobi, x_seidel, k_seidel, x_ref,
                         res_jacobi, res_seidel, err_jacobi, err_seidel)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print("Отчет записан в файл:", REPORT_PATH)


if __name__ == "__main__":
    main()