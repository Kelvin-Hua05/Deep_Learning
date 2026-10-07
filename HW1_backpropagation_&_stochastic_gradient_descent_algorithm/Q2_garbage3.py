import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

DATA_PATH = "2026_ionosphere_data.csv"

# ====== 全域設定 ======
LATENT_DIMS = [2, 5, 10]
SNAPSHOTS_AT = [10, 50, 100, 200]
MAIN_SEED = 42
N_SEEDS = 10             # 最終對照的次數（想快速測試可改 3）
CV_FOLDS = 3
BATCH = 32
USE_ENSEMBLE = True      # 額外討論用；不需要可關掉以節省時間
ENSEMBLE_K = 5

# 原始 baseline：SGD、無正則化、ReLU latent
BASE = dict(hidden=[32, 16], drop=0.0, l2=0.0, lr=0.05, mom=0.0,
            act="relu", cosine=False, epochs=300)

# 改良版：SGD+momentum、cosine 學習率衰減、tanh latent、dropout、L2
COMMON = dict(lr=0.02, mom=0.9, act="tanh", cosine=True, epochs=200)
GRID = [{**COMMON, **g} for g in [
    dict(hidden=[32, 16], drop=0.0, l2=1e-2),
    dict(hidden=[32, 16], drop=0.2, l2=1e-2),
    dict(hidden=[64, 32], drop=0.3, l2=1e-2),
    dict(hidden=[64, 32], drop=0.3, l2=3e-2),
]]


def cfg_str(c):
    return f"hidden={c['hidden']}, dropout={c['drop']}, L2={c['l2']}"


# ====== 資料 ======
df = pd.read_csv(DATA_PATH, header=None)
X_all = df.iloc[:, :-1].to_numpy(dtype=np.float64)
y_all = (df.iloc[:, -1].to_numpy() == "g").astype(int)   # g=1, b=0
print("資料大小:", df.shape, " g:", (y_all == 1).sum(), " b:", (y_all == 0).sum())


def stratified_split(y, frac, rng):
    """每個類別各自抽 frac 比例，回傳 (剩下的 idx, 抽出的 idx)"""
    rest, picked = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        k = int(round(len(idx) * frac))
        picked.append(idx[:k])
        rest.append(idx[k:])
    return rng.permutation(np.concatenate(rest)), rng.permutation(np.concatenate(picked))


def make_data(seed):
    """分層 80/20 切分；標準化只用 train 的統計量"""
    rng = np.random.RandomState(seed)
    tr, te = stratified_split(y_all, 0.2, rng)
    mu, sd = X_all[tr].mean(0), X_all[tr].std(0)
    sd[sd == 0] = 1.0
    f = lambda i: (X_all[i] - mu) / sd
    return dict(Xtr=f(tr), ytr=y_all[tr], Xte=f(te), yte=y_all[te])


# ====== 模型 ======
class MLP:
    """隱藏層 ReLU(+dropout) -> latent 層(tanh 或 ReLU) -> Softmax 輸出"""

    def __init__(self, sizes, cfg):
        self.cfg = cfg
        self.L = len(sizes) - 1
        self.lr, self.l2, self.mom = cfg["lr"], cfg["l2"], cfg["mom"]
        self.p, self.act = cfg["drop"], cfg["act"]
        self.W, self.b = [], []
        for i in range(self.L):
            is_tanh_latent = (i == self.L - 2 and self.act == "tanh")
            scale = np.sqrt((1.0 if is_tanh_latent else 2.0) / sizes[i])
            self.W.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.b.append(np.zeros((1, sizes[i + 1])))
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    @staticmethod
    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def forward(self, X, train=False):
        a = X
        self.z, self.a, self.mask = [], [X], []
        for i in range(self.L):
            z = a @ self.W[i] + self.b[i]
            m = None
            if i == self.L - 1:
                a = self.softmax(z)
            elif i == self.L - 2:                       # latent 層：不做 dropout
                a = np.tanh(z) if self.act == "tanh" else np.maximum(0, z)
            else:
                a = np.maximum(0, z)
                if train and self.p > 0:
                    m = (np.random.rand(*a.shape) > self.p) / (1.0 - self.p)
                    a = a * m
            self.z.append(z)
            self.a.append(a)
            self.mask.append(m)
        return a

    def backward(self, Y):
        m = Y.shape[0]
        dZ = (self.a[-1] - Y) / m                       # softmax + CE
        gW, gb = [None] * self.L, [None] * self.L
        for i in reversed(range(self.L)):
            gW[i] = self.a[i].T @ dZ + self.l2 * self.W[i]
            gb[i] = dZ.sum(0, keepdims=True)
            if i > 0:
                dA = dZ @ self.W[i].T
                k = i - 1
                if self.mask[k] is not None:
                    dA = dA * self.mask[k]
                if k == self.L - 2 and self.act == "tanh":
                    dZ = dA * (1.0 - self.a[k + 1] ** 2)
                else:
                    dZ = dA * (self.z[k] > 0)
        for i in range(self.L):
            self.vW[i] = self.mom * self.vW[i] - self.lr * gW[i]
            self.vb[i] = self.mom * self.vb[i] - self.lr * gb[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    def latent(self, X):
        self.forward(X)
        return self.a[-2]


def ce(p, y, eps=1e-9):
    return -np.mean(np.log(p[np.arange(len(y)), y] + eps))


def err(p, y):
    return np.mean(p.argmax(1) != y)


def train(d, dim, seed, cfg, snapshots=(), track=True):
    """回傳 model, 歷史紀錄, latent 快照, 最終結果"""
    np.random.seed(seed)
    model = MLP([34] + cfg["hidden"] + [dim, 2], cfg)
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    Ytr = np.eye(2)[ytr]
    n, E = len(ytr), cfg["epochs"]
    H = dict(loss=[], te_loss=[], tr=[], te=[])
    snaps = {}

    for ep in range(1, E + 1):
        if cfg["cosine"]:
            model.lr = cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (ep - 1) / E)))
        perm = np.random.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            model.forward(Xtr[idx], train=True)
            model.backward(Ytr[idx])
        if track:
            ptr, pte = model.forward(Xtr), model.forward(Xte)
            H["loss"].append(ce(ptr, ytr))
            H["te_loss"].append(ce(pte, yte))
            H["tr"].append(err(ptr, ytr))
            H["te"].append(err(pte, yte))
        if ep in snapshots:
            snaps[ep] = (model.latent(Xtr), model.latent(Xte),
                         model.W[-1].copy(), model.b[-1].copy(),
                         err(model.forward(Xtr), ytr), err(model.forward(Xte), yte))

    res = dict(train_err=err(model.forward(Xtr), ytr),
               test_err=err(model.forward(Xte), yte))
    return model, H, snaps, res


def cv_select(d, seed):
    """只用訓練資料做 stratified k-fold CV，挑出最佳超參數組合（不碰測試集）"""
    X, y = d["Xtr"], d["ytr"]
    rng = np.random.RandomState(seed + 1000)
    folds = [[] for _ in range(CV_FOLDS)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for k, part in enumerate(np.array_split(idx, CV_FOLDS)):
            folds[k].extend(part.tolist())
    folds = [np.array(f) for f in folds]

    scores = []
    for cfg in GRID:
        errs, ces = [], []
        for k in range(CV_FOLDS):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(CV_FOLDS) if j != k])
            dk = dict(Xtr=X[tr], ytr=y[tr], Xte=X[va], yte=y[va])
            m, _, _, r = train(dk, 5, seed * 10 + k, cfg, track=False)
            errs.append(r["test_err"])
            ces.append(ce(m.forward(X[va]), y[va]))
        scores.append((np.mean(errs), np.mean(ces)))
    best = min(range(len(GRID)), key=lambda i: (round(scores[i][0], 4), scores[i][1]))
    return best, scores


def ensemble_test_err(d, dim, seed, cfg, K=ENSEMBLE_K):
    P = 0
    for k in range(K):
        m, *_ = train(d, dim, seed * 100 + k, cfg, track=False)
        P = P + m.forward(d["Xte"])
    return err(P / K, d["yte"])


# ====== 視覺化：含測試點與決策邊界 ======
def plot_latent(ax, snap, ytr, yte, dim, title, legend=False):
    Ztr, Zte, Wl, bl, _, _ = snap
    w = Wl[:, 1] - Wl[:, 0]            # logit(g) - logit(b) = z @ w + c
    c = bl[0, 1] - bl[0, 0]
    if dim == 2:
        Ptr, Pte = Ztr, Zte
        xl, yl = "latent unit 1", "latent unit 2"
    else:
        wn = w / (np.linalg.norm(w) + 1e-12)
        Rtr = Ztr - np.outer(Ztr @ wn, wn)          # 去掉決策方向後的殘差
        Rte = Zte - np.outer(Zte @ wn, wn)
        mu = Rtr.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Rtr - mu, full_matrices=False)
        Ptr = np.c_[Ztr @ w + c, (Rtr - mu) @ Vt[0]]
        Pte = np.c_[Zte @ w + c, (Rte - mu) @ Vt[0]]
        xl, yl = "logit(g) - logit(b)", "PC1 (orthogonal to decision axis)"

    for cls, col, name in [(0, "navy", "b"), (1, "green", "g")]:
        ax.scatter(Ptr[ytr == cls, 0], Ptr[ytr == cls, 1], c=col, s=12, alpha=0.45,
                   label=f"{name} train")
        ax.scatter(Pte[yte == cls, 0], Pte[yte == cls, 1], c=col, s=40, marker="^",
                   edgecolors="k", linewidths=0.6, label=f"{name} test")

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if dim == 2:
        if abs(w[1]) > 1e-9:
            xs = np.array([x0, x1])
            ax.plot(xs, -(w[0] * xs + c) / w[1], "r--", lw=1.3, label="decision boundary")
        elif abs(w[0]) > 1e-9:
            ax.axvline(-c / w[0], color="r", ls="--", lw=1.3, label="decision boundary")
    else:
        ax.axvline(0, color="r", ls="--", lw=1.3, label="decision boundary")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    if legend:
        ax.legend(fontsize=7, loc="best")


# ====== (a)(b) 主模型 ======
d = make_data(MAIN_SEED)
best_i, cv_scores = cv_select(d, MAIN_SEED)
cfg = GRID[best_i]
print("\n[CV 超參數選擇]（只用訓練資料）")
for i, (e_, l_) in enumerate(cv_scores):
    mark = "  <- 選中" if i == best_i else ""
    print(f"  {cfg_str(GRID[i])}: CV err={e_:.4f}, CV loss={l_:.4f}{mark}")

model, H, _, res = train(d, 2, MAIN_SEED, cfg)
print(f"\n[主模型] 架構: 34-{'-'.join(map(str, cfg['hidden']))}-2-2 "
      f"(隱藏層 ReLU, latent {cfg['act']}, 輸出 Softmax)")
print(f"lr={cfg['lr']} (cosine), momentum={cfg['mom']}, dropout={cfg['drop']}, "
      f"L2={cfg['l2']}, epochs={cfg['epochs']}")
print(f"Train samples={len(d['ytr'])}, Test samples={len(d['yte'])}")
print(f"Train error = {res['train_err']:.4f}   Test error = {res['test_err']:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(H["loss"], label="Train loss")
ax[0].plot(H["te_loss"], label="Test loss")
ax[0].set_title("Learning Curve (Cross-Entropy Loss)")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[1].plot(H["tr"], label="Train error")
ax[1].plot(H["te"], label="Test error")
ax[1].set_title("Error Rate")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig("learning_curve_classification.png", dpi=150)
plt.show()

# ====== (c) 不同 latent_dim × 不同訓練階段 ======
fig, axes = plt.subplots(len(LATENT_DIMS), len(SNAPSHOTS_AT),
                         figsize=(4.6 * len(SNAPSHOTS_AT), 4.0 * len(LATENT_DIMS)))
for r, dim in enumerate(LATENT_DIMS):
    _, _, snaps, rs = train(d, dim, MAIN_SEED, cfg, snapshots=SNAPSHOTS_AT)
    print(f"latent_dim={dim}: train_err={rs['train_err']:.4f}, test_err={rs['test_err']:.4f}")
    for c_, ep in enumerate(SNAPSHOTS_AT):
        s = snaps[ep]
        plot_latent(axes[r, c_], s, d["ytr"], d["yte"], dim,
                    f"dim={dim}, epoch={ep}\ntrain err={s[4]:.3f}, test err={s[5]:.3f}",
                    legend=(r == 0 and c_ == 0))
plt.tight_layout()
plt.savefig("compare_latent_dims.png", dpi=150)
plt.show()

# ====== 最終對照：每個 seed 各自做 CV 選超參數（無測試集洩漏）======
print(f"\n===== 最終對照（{N_SEEDS} 次不同切分）=====")
res_b = {k: [] for k in LATENT_DIMS}
res_i = {k: [] for k in LATENT_DIMS}
res_t = {k: [] for k in LATENT_DIMS}
res_e = {k: [] for k in LATENT_DIMS}
picked = Counter()

for s in range(N_SEEDS):
    ds = make_data(s)
    bi, _ = cv_select(ds, s)
    picked[bi] += 1
    for dim in LATENT_DIMS:
        res_b[dim].append(train(ds, dim, s, BASE, track=False)[3]["test_err"])
        r_ = train(ds, dim, s, GRID[bi], track=False)[3]
        res_i[dim].append(r_["test_err"])
        res_t[dim].append(r_["train_err"])
        if USE_ENSEMBLE:
            res_e[dim].append(ensemble_test_err(ds, dim, s, GRID[bi]))
    print(f"  seed {s} 完成（CV 選中第 {bi} 組）")

print("\nCV 選中的超參數次數：")
for i, n_ in sorted(picked.items()):
    print(f"  {cfg_str(GRID[i])}: {n_} 次")

print(f"\n{'dim':>4} | {'baseline':>15} | {'improved':>15} | "
      f"{'paired diff (base-imp)':>24} | {'ensemble':>15} | train err")
for dim in LATENT_DIMS:
    b, i_, t = map(np.array, (res_b[dim], res_i[dim], res_t[dim]))
    diff = b - i_
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    ens = (f"{np.mean(res_e[dim]):.4f} ± {np.std(res_e[dim]):.4f}"
           if USE_ENSEMBLE else "-")
    print(f"{dim:>4} | {b.mean():.4f} ± {b.std():.4f} | {i_.mean():.4f} ± {i_.std():.4f} | "
          f"{diff.mean():+.4f} ± {se:.4f} (SE)   | {ens} | {t.mean():.4f}")
print("\n（paired diff 為正代表改良版較好；若 |diff| 小於 2 倍 SE，改善不顯著）")
