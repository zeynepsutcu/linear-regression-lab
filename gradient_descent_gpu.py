import math
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

x_train = torch.tensor([1.0, 2.0], device=device)
y_train = torch.tensor([300.0, 500.0], device=device)


def compute_cost(x, y, w, b):
    f_wb = w * x + b
    return ((f_wb - y) ** 2).mean() / 2


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = torch.tensor(w_in, device=device, requires_grad=True)
    b = torch.tensor(b_in, device=device, requires_grad=True)
    J_history = []
    p_history = []

    for i in range(num_iters):
        cost = compute_cost(x, y, w, b)
        cost.backward()

        with torch.no_grad():
            w -= alpha * w.grad
            b -= alpha * b.grad
            w.grad.zero_()
            b.grad.zero_()

        if i < 100000:
            J_history.append(cost.item())
            p_history.append([w.item(), b.item()])
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} "
                  f"w: {w.item(): 0.3e}, b:{b.item(): 0.5e}")

    return w.item(), b.item(), J_history, p_history


w_init = 0.0
b_init = 0.0
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + torch.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")
