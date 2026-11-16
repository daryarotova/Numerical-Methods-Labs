def read_data(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = [ln.strip() for ln in file if ln.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")

    if len(lines) < 2:
        raise ValueError(f"В файле должно быть 2 строки, найдено: {len(lines)}")

    try:
        x_nodes = [float(v) for v in lines[0].split()]
        y_nodes = [float(v) for v in lines[1].split()]
    except ValueError as e:
        raise ValueError(f"Неверный формат числа в файле: {e}")

    if len(x_nodes) != len(y_nodes):
        raise ValueError("Количество X и Y должно совпадать")
    
    return x_nodes, y_nodes


def evaluate(coeffs, x):
    return sum(c * (x ** i) for i, c in enumerate(coeffs))


def solve_gauss(A, B):
    n = len(B)
    a = [row[:] for row in A]
    b = B[:]
    for i in range(0, n):
        for j in range(i+1, n):
            c = a[j][i] / a[i][i]
            for k in range(i, n):
                a[j][k] -= c * a[i][k]
            b[j] -= c * b[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = 0.0
        for j in range(i+1, n):
            sum_ax += a[i][j] * x[j]
        x[i] = (b[i] - sum_ax) / a[i][i]
    return x


def build_system(x_nodes, y_nodes, degree):
    n = degree
    A = [[0.0] * (n + 1) for _ in range(n + 1)]
    B = [0.0] * (n + 1)

    for i in range(n + 1):
        for j in range(n + 1):
            power = i + j
            A[i][j] = sum(x_i ** power for x_i in x_nodes)
        B[i] = sum(y_nodes[x_k] * (x_nodes[x_k] ** i) for x_k in range(len(x_nodes)))

    a_coeffs = solve_gauss(A, B)
    return a_coeffs


def calculate_mse(x_nodes, y_nodes, coeffs):
    total_error = 0.0
    for x_j, y_j in zip(x_nodes, y_nodes):
        f_x = evaluate(coeffs, x_j)
        diff = f_x - y_j
        total_error += diff ** 2
    return total_error


def format_polynomial(coeffs):
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if abs(c) < 1e-10 and i != 0:
            continue
        if i == 0:
            var = ""
        elif i == 1:
            var = "x"
        else:
            var = f"x^{i}"

        if len(terms) == 0:
            sign = "-" if c < 0 else ""
            terms.append(f"{sign}{abs(c):.6f}{var}")
        else:
            sign = "-" if c < 0 else "+"
            terms.append(f"{sign} {abs(c):.6f}{var}")

    return " ".join(terms).replace("+ -", "- ")
