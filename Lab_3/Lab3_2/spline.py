def read_data(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = [ln.strip() for ln in file if ln.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")

    if len(lines) < 3:
        raise ValueError(f"В файле должно быть 3 строки, найдено: {len(lines)}")

    try:
        x_nodes = [float(v) for v in lines[0].split()]
        y_nodes = [float(v) for v in lines[1].split()]
        x_given = float(lines[2])
    except ValueError as e:
        raise ValueError(f"Неверный формат числа в файле: {e}")

    if len(x_nodes) != len(y_nodes):
        raise ValueError("Количество X и Y должно совпадать")
    if len(x_nodes) < 4:
        raise ValueError("Нужно хотя бы 4 узла")

    return x_nodes, y_nodes, x_given


def count_steps(x_nodes):
    h = [x_nodes[i] - x_nodes[i-1] for i in range(1, len(x_nodes))]
    return h


def sweep_forward(a, b, c, d):
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i-1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i-1]) / denom
    
    return P, Q


def sweep_backward(P, Q):
    n = len(Q)
    x = [0.0] * n
    x[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i] * x[i+1] + Q[i]
    return x


def find_c_coefficients(h, y_nodes):
    a = [0.0, h[1], h[2]]
    b = [2*(h[0]+h[1]), 2*(h[1]+h[2]), 2*(h[2]+h[3])]
    c = [h[1], h[2], 0.0]
    d = [
        3 * ((y_nodes[2]-y_nodes[1])/h[1] - (y_nodes[1]-y_nodes[0])/h[0]),
        3 * ((y_nodes[3]-y_nodes[2])/h[2] - (y_nodes[2]-y_nodes[1])/h[1]),
        3 * ((y_nodes[4]-y_nodes[3])/h[3] - (y_nodes[3]-y_nodes[2])/h[2])
    ]

    P, Q = sweep_forward(a, b, c, d)
    c_inner = sweep_backward(P, Q)
    c_full = [0.0] + c_inner + [0.0]
    return c_full


def build_spline_coeffs(x_nodes, y_nodes, h, c_full):
    coeffs = []
    for i in range(len(x_nodes) - 1):
        a = y_nodes[i]
        c_i = c_full[i]
        c_next = c_full[i+1]
        h_i = h[i]

        if i < len(h) - 1:
            b = ((y_nodes[i+1] - y_nodes[i]) / h_i) - (1/3) * h[i] * (c_next + 2 * c_i)
            d = (c_next - c_i) / (3 * h_i)
        
        if i == len(h) - 1:
            b = ((y_nodes[i+1] - y_nodes[i]) / h_i) - (2/3) * h_i * c_i
            d = - c_i / (3 * h_i)
        
        coeffs.append((a, b, c_i, d))
    
    return coeffs


def evaluate_spline(x_given, x_nodes, coeffs):
    if x_given < x_nodes[0] or x_given > x_nodes[-1]:
        raise ValueError("Точка находится вне отрезка.")
    
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_given <= x_nodes[i+1]:
            a, b, c, d = coeffs[i]
            bracket = x_given - x_nodes[i]
            s_x_given = a + b * bracket + c * (bracket ** 2) + d * (bracket ** 3)
            return s_x_given, i+1
        
    raise ValueError("Не удалось определить интервал для заданной точки.")

