import numpy as np
from checks import check_convergence_condition, check_diagonal_dominance, check_solution, solve_reference
from solve import jacobi_solve, seidel_solve, build_equivalent_form
from utils import read_iteration_system_from_files
from report_utils import add_section, add_matrix, add_value, add_list


MATRIX_A_PATH = "data/matrix_A.txt"
VECTOR_B_PATH = "data/vector_b.txt"
EPSILON_PATH = "data/epsilon.txt"
REPORT_PATH = "data/report_lab_1_3.txt"


def make_report(A, b, eps, alpha, beta, norm_alpha, converged, diag_dominance,
                x_jacobi, k_jacobi, x_seidel, k_seidel, x_ref,
                res_jacobi, res_seidel, err_jacobi, err_seidel):
    lines = []
    lines.append("Лабораторная работа 1.3: метод простых итераций и метод Зейделя")
    lines.append("")
    add_value(lines, "Размерность системы", A.shape[0])

    add_section(lines, "Входные данные")
    add_matrix(lines, "Матрица A", A)
    add_matrix(lines, "Вектор b", b)
    add_value(lines, "Точность epsilon", eps)

    add_section(lines, "Эквивалентная форма")
    add_matrix(lines, "Матрица alpha", alpha)
    add_matrix(lines, "Вектор beta", beta)
    add_value(lines, "Норма матрицы alpha ||alpha||_c", norm_alpha)

    add_section(lines, "Проверка условий сходимости")
    add_value(lines, "Условие сходимости (||alpha||_c < 1)", converged)
    add_value(lines, "Диагональное преобладание матрицы A", diag_dominance)

    add_section(lines, "Метод простых итераций (Якоби)")
    add_matrix(lines, "Решение x", x_jacobi)
    add_value(lines, "Число итераций", k_jacobi)
    add_value(lines, "Невязка ||A @ x - b||_c", f"{res_jacobi:.3e}")
    add_value(lines, "Отклонение от эталона ||x - x_ref||_c", f"{err_jacobi:.3e}")

    add_section(lines, "Метод Зейделя")
    add_matrix(lines, "Решение x", x_seidel)
    add_value(lines, "Число итераций", k_seidel)
    add_value(lines, "Невязка ||A @ x - b||_c", f"{res_seidel:.3e}")
    add_value(lines, "Отклонение от эталона ||x - x_ref||_c", f"{err_seidel:.3e}")

    add_section(lines, "Эталонное решение")
    add_matrix(lines, "x_ref", x_ref)

    add_section(lines, "Сравнение методов")
    add_value(lines, "Число итераций метода Якоби", k_jacobi)
    add_value(lines, "Число итераций метода Зейделя", k_seidel)
    add_value(lines, "Зейдель быстрее Якоби (во столько раз)", f"{k_jacobi / k_seidel:.2f}")

    return "\n".join(lines)


def main():
    A, b, epsilon = read_iteration_system_from_files(MATRIX_A_PATH, VECTOR_B_PATH, EPSILON_PATH)
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