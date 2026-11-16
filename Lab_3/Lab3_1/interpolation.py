import math


def read_data(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = [ln.strip() for ln in file if ln.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")

    if len(lines) < 3:
        raise ValueError(f"В файле должно быть 3 строки, найдено: {len(lines)}")

    try:
        x_nodes_a = [float(v) for v in lines[0].split()]
        x_nodes_b = [float(v) for v in lines[1].split()]
        x_given = float(lines[2])
    except ValueError as e:
        raise ValueError(f"Неверный формат числа в файле: {e}")

    if len(x_nodes_a) != 4 or len(x_nodes_b) != 4:
        raise ValueError("Ожидалось по 4 узла в каждой строке")

    return x_nodes_a, x_nodes_b, x_given


def function(x):
    return math.sqrt(x) + x


def lagrange_evaluate(x_nodes, y_nodes, x_given):
    total_sum = 0.0
    for i in range(0, len(x_nodes)):
        product = 1.0
        for j in range(0, len(x_nodes)):
            if i == j:
                continue
            product *= ((x_given - x_nodes[j]) / (x_nodes[i] - x_nodes[j]))
        total_sum += y_nodes[i] * product
    return total_sum


def table_of_divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    tbl = [[0.0] * n for _ in range(n)]

    for i in range(n):
        tbl[i][0] = y_nodes[i]

    for k in range(1, n):
        for j in range(0, n - k):
            tbl[j][k] = (tbl[j+1][k-1] - tbl[j][k-1]) / (x_nodes[j+k] - x_nodes[j])
    
    return tbl


def newton_evaluate(x_nodes, table, x_given):
    n = len(x_nodes)
    result = table[0][0]
    product = 1.0
    for k in range(1, n):
        product *= (x_given - x_nodes[k - 1])
        result += table[0][k] * product
    return result


def _factor(x):
    if x > 0:
        return f"(x - {x:g})"
    if x < 0:
        return f"(x + {abs(x):g})"
    return "x"


def _join_terms(parts):
    result = ""
    for coeff, factors in parts:
        if result == "":
            result = f"{coeff:.5f}{factors}"
        else:
            sign = "+" if coeff >= 0 else "-"
            result += f" {sign} {abs(coeff):.5f}{factors}"
    return result


def lagrange_polynomial_string(x_nodes, y_nodes):
    n = len(x_nodes)
    parts = []
    for i in range(n):
        denom = 1.0
        for j in range(n):
            if i != j:
                denom *= (x_nodes[i] - x_nodes[j])
        coeff = y_nodes[i] / denom
        if abs(coeff) < 1e-12:
            continue
        factors = "".join(_factor(x_nodes[j]) for j in range(n) if j != i)
        parts.append((coeff, factors))
    return f"L_{n-1}(x) = " + _join_terms(parts)


def newton_polynomial_string(x_nodes, table):
    n = len(x_nodes)
    parts = [(table[0][0], "")]
    factors = ""
    for k in range(1, n):
        factors += _factor(x_nodes[k - 1])
        parts.append((table[0][k], factors))
    return f"P_{n-1}(x) = " + _join_terms(parts)
