import numpy as np


def sweep_forward(a, b, c, d):
    n = b.shape[0]
    P = np.zeros(n)
    Q = np.zeros(n)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] + a[i] * P[i - 1]
        P[i] = (-c[i]) / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator
    
    return P, Q


def sweep_backward(P, Q):
    n = Q.shape[0]
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


if __name__ == '__main__':
    a = np.array([0., -1., 2., -1.])
    b = np.array([8., 6., 10., 6.])
    c = np.array([-2., -2., -4., 0.])
    d = np.array([6., 3., 8., 5.])
    P, Q = sweep_forward(a, b, c, d)
    x = sweep_backward(P, Q)
    x_true = np.array([1., 1., 1., 1.])
    check = np.allclose(x, x_true)
    print("P = ", P)
    print("Q = ", Q)
    print("x = ", x)
    print("check:", check)