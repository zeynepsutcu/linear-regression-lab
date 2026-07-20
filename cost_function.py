import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use('./deeplearning.mplstyle')
except OSError:
    pass

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


w = 200
b = 100
print(f"Cost at w={w}, b={b}: {compute_cost(x_train, y_train, w, b)}")

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

w = 209
b = 2.4
print(f"Cost at w={w}, b={b}: {compute_cost(x_train, y_train, w, b)}")
