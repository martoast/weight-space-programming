import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=160)
D = np.load("data.npz"); X, y = D["Xtr"], D["ytr"]; ys = y*2-1
P = dict(np.load("broken_1.npz")); W1, b1, W2 = P["W1"], P["b1"], P["W2"][:, 0]
act = (X @ W1 + b1) > 0
# Net's average sensitivity to each input: sum_j W2_j W1_ij * P(unit j on), computed from weights
sens = W1 @ (W2 * act.mean(0))
# What the data says: label correlation of each input, and of each input's |x| (nonlinear part)
c_lin = np.array([np.corrcoef(X[:, i], ys)[0, 1] for i in range(20)])
print("net sensitivity (scaled):", sens / np.abs(sens).max())
print("data corr(x_i, y)  scaled:", c_lin / np.abs(c_lin).max())
r = (sens/np.abs(sens).max()) - (c_lin/np.abs(c_lin).max())
print("mismatch:", r)
