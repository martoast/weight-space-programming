import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=160)
D = np.load("data.npz"); X, y = D["Xtr"], D["ytr"]
for i in range(3):
    P = dict(np.load(f"broken_{i}.npz"))
    W1, b1, W2, b2 = P["W1"], P["b1"], P["W2"][:, 0], P["b2"]
    pre = X @ W1 + b1
    print(f"\n##### broken_{i}")
    print("W2 (out weights):", W2)
    print("|W1| col norms:  ", np.linalg.norm(W1, axis=0))
    print("b1:              ", b1)
    print("unit active rate:", (pre > 0).mean(0))
    print("|W1| row norms (per input):", np.linalg.norm(W1, axis=1))
    C = np.corrcoef(W1.T); np.fill_diagonal(C, 0)
    j, k = np.unravel_index(np.abs(C).argmax(), C.shape)
    print(f"max |corr| between unit input-weights: units {j},{k} = {C[j,k]:.3f}")
    # sign agreement: does unit's activation correlate with label in the direction W2 says?
    h = np.maximum(pre, 0); ys = y * 2 - 1
    corr = np.array([np.corrcoef(h[:, u], ys)[0, 1] if h[:, u].std() > 0 else 0 for u in range(len(W2))])
    print("sign(W2)*corr(h,label):", np.sign(W2) * corr)
