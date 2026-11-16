def function(x):
    return x / (16 - x ** 4)


def generate_grid(a, b, h):
    n = round((b - a) / h)
    x_nodes = [a + i * h for i in range(n + 1)]
    return x_nodes, n


def left_rectangles(x_nodes, h):
    result = 0.0
    n = len(x_nodes)
    for i in range(n - 1):
        result += function(x_nodes[i])
    return result * h


def right_rectangles(x_nodes, h):
    result = 0.0
    n = len(x_nodes)
    for i in range(1, n):
        result += function(x_nodes[i])
    return result * h


def midpoint_rectangles(x_nodes, h):
    result = 0.0
    n = len(x_nodes)
    for i in range(n - 1):
        x_mid = (x_nodes[i] + x_nodes[i+1]) / 2
        result += function(x_mid)
    return result * h


def trapezoidal(x_nodes, h):
    n = len(x_nodes)
    result = function(x_nodes[0]) + function(x_nodes[-1])
    for i in range(1, n - 1):
        result += 2 * function(x_nodes[i])
    return result * h / 2


def simpson(x_nodes, h):
    n = len(x_nodes)
    n_intervals = n - 1
    if n_intervals % 2 != 0:
        raise ValueError("Количество интервалов должно быть четным для метода Симпсона")
    
    result = function(x_nodes[0]) + function(x_nodes[-1])
    for i in range(1, n - 1):
        if i % 2 == 1:
            result += 4 * function(x_nodes[i])
        else:
            result += 2 * function(x_nodes[i])
    return result * h / 3


def runge_romberg(I_h1, I_h2, p):
    return I_h2 + (I_h2 - I_h1) / (2.0 ** p - 1)
