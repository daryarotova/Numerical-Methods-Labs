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
    if len(x_nodes) < 5:
        raise ValueError("Нужно хотя бы 5 узлов")

    return x_nodes, y_nodes, x_given


def table_of_divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    tbl = [[0.0] * n for _ in range(n)]

    for i in range(n):
        tbl[i][0] = y_nodes[i]

    for k in range(1, n):
        for j in range(0, n - k):
            tbl[j][k] = (tbl[j+1][k-1] - tbl[j][k-1]) / (x_nodes[j+k] - x_nodes[j])
    
    return tbl


def calculate_derivatives_newton(x_nodes, table, x_given):
    d1 = d2 = 0.0
    Q = 1.0
    Q_first = 0.0
    Q_second = 0.0
    n = len(x_nodes)

    for k in range(n):
        c_k = table[0][k]
        d1 += c_k * Q_first
        d2 += c_k * Q_second
        if k < n - 1:
            Q_new = Q * (x_given - x_nodes[k])
            Q_first_new = Q_first * (x_given - x_nodes[k]) + Q
            Q_second_new = Q_second * (x_given - x_nodes[k]) + 2 * Q_first
            Q, Q_first, Q_second = Q_new, Q_first_new, Q_second_new
    
    return d1, d2


def calculate_derivatives_local(x_nodes, y_nodes, x_given):
    k = -1
    for i in range(len(x_nodes)):
        if abs(x_nodes[i] - x_given) < 1e-10:
            k = i
            break
            
    if k <= 0 or k >= len(x_nodes) - 1:
        raise ValueError("Точка должна совпадать с внутренним узлом сетки.")

    h = x_nodes[k] - x_nodes[k-1]
    y_k = y_nodes[k]
    y_prev = y_nodes[k-1]
    y_next = y_nodes[k+1]

    d1_left = (y_k - y_prev) / h
    d1_right = (y_next - y_k) / h
    d1_central = (y_next - y_prev) / (2 * h)
    d2 = (y_next - 2 * y_k + y_prev) / (h ** 2)

    return d1_left, d1_right, d1_central, d2