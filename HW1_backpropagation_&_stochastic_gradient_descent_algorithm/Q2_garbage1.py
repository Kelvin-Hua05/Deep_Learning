# %% [Cell 1] Imports & 基本設定
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# %% [Cell 2] 讀取 Ionosphere 資料
# 注意：這份 Excel 檔案「沒有標題列」，所以一定要用 header=None，
# 否則會把第一筆資料誤當成欄位名稱，少算一筆樣本。
DATA_PATH = "2026_ionosphere_data.csv"

df = pd.read_csv(DATA_PATH, header=None)
print("資料大小:", df.shape)   # 應該是 (351, 35)

X_raw = df.iloc[:, :-1].to_numpy(dtype=np.float64)   # 34 個特徵
labels = df.iloc[:, -1].to_numpy()                    # 'g' / 'b'

# 標籤轉成 one-hot: g -> [0,1], b -> [1,0]（類別順序可自行調整）
classes = np.array(["b", "g"])
y_idx = (labels == "g").astype(int)          # g=1, b=0
Y_onehot = np.eye(2)[y_idx]                   # (N, 2)

print("g 數量:", (y_idx == 1).sum(), " b 數量:", (y_idx == 0).sum())

# %% [Cell 3] 特徵標準化 + 切分 80% train / 20% test
X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
X_std[X_std == 0] = 1.0   # 避免除以 0（例如全部是 0 的那個常數欄位）
X_norm = (X_raw - X_mean) / X_std

N = X_norm.shape[0]
perm = np.random.permutation(N)
X_norm, Y_onehot, y_idx = X_norm[perm], Y_onehot[perm], y_idx[perm]

n_train = int(N * 0.8)
X_train, Y_train, y_train_idx = X_norm[:n_train], Y_onehot[:n_train], y_idx[:n_train]
X_test, Y_test, y_test_idx = X_norm[n_train:], Y_onehot[n_train:], y_idx[n_train:]
print("Train:", X_train.shape, " Test:", X_test.shape)

# %% [Cell 4] 手刻 MLP（分類版）：hidden 用 ReLU，輸出層用 Softmax + Cross-Entropy
# 架構設計成: [輸入34] -> 隱藏層們 -> [latent_dim] (輸出層前的那一層) -> [2] (softmax 輸出)
# latent_dim 就是作業 (c) 要求比較的「輸出層前一層的節點數」
class MLP:
    def __init__(self, layer_sizes, lr=0.05):
        """
        layer_sizes 範例: [34, 32, 16, latent_dim, 2]
        最後一個數字固定是 2（二分類）；倒數第二個數字就是 latent_dim。
        """
        self.lr = lr
        self.L = len(layer_sizes) - 1
        self.W, self.b = [], []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i + 1]))
            self.W.append(w)
            self.b.append(b)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)  # 數值穩定
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        a = X
        self.z_cache = []
        self.a_cache = [a]
        for i in range(self.L):
            z = a @ self.W[i] + self.b[i]
            if i < self.L - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            self.z_cache.append(z)
            self.a_cache.append(a)
        return a

    def backward(self, Y):
        m = Y.shape[0]
        y_pred = self.a_cache[-1]
        # softmax + cross-entropy 梯度化簡後就是 (y_pred - Y) / m
        dZ = (y_pred - Y) / m

        grads_W = [None] * self.L
        grads_b = [None] * self.L
        for i in reversed(range(self.L)):
            A_prev = self.a_cache[i]
            grads_W[i] = A_prev.T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dZ @ self.W[i].T
                dZ = dA_prev * self.relu_grad(self.z_cache[i - 1])

        for i in range(self.L):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def predict_proba(self, X):
        return self.forward(X)

    def latent(self, X):
        """回傳輸出層「前一層」的 activation（也就是作業要求畫分布圖的那一層）"""
        self.forward(X)
        return self.a_cache[-2]

def cross_entropy(y_pred, Y_onehot, eps=1e-9):
    return -np.mean(np.sum(Y_onehot * np.log(y_pred + eps), axis=1))

def error_rate(y_pred, y_true_idx):
    pred_idx = np.argmax(y_pred, axis=1)
    return np.mean(pred_idx != y_true_idx)

# %% [Cell 5] 訓練函式：mini-batch SGD，並在指定 epoch 記錄 latent feature
def train_model(X_train, Y_train, y_train_idx, X_test, Y_test, y_test_idx,
                 layer_sizes, lr, epochs, batch_size, snapshot_epochs):
    model = MLP(layer_sizes, lr=lr)
    n = X_train.shape[0]
    loss_history, train_err_history, test_err_history = [], [], []
    latent_snapshots = {}

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)
        Xs, Ys = X_train[perm], Y_train[perm]
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = Ys[start:start + batch_size]
            pred = model.forward(xb)
            epoch_loss += cross_entropy(pred, yb) * xb.shape[0]
            model.backward(yb)
        loss_history.append(epoch_loss / n)

        train_pred = model.predict_proba(X_train)
        test_pred = model.predict_proba(X_test)
        train_err_history.append(error_rate(train_pred, y_train_idx))
        test_err_history.append(error_rate(test_pred, y_test_idx))

        if epoch in snapshot_epochs:
            latent_snapshots[epoch] = model.latent(X_train)
            print(f"Epoch {epoch:4d} | loss={loss_history[-1]:.4f} "
                  f"| train_err={train_err_history[-1]:.3f} | test_err={test_err_history[-1]:.3f}")

    return model, loss_history, train_err_history, test_err_history, latent_snapshots

# %% [Cell 6] 訓練主模型（latent_dim = 2，可以直接畫 2D 分布圖）
EPOCHS = 1000
SNAPSHOTS_AT = [10, 100, 500, 1000]
LATENT_DIM = 2

model, loss_hist, train_err_hist, test_err_hist, latent_snap = train_model(
    X_train, Y_train, y_train_idx, X_test, Y_test, y_test_idx,
    layer_sizes=[34, 32, 16, LATENT_DIM, 2],
    lr=0.1, epochs=EPOCHS, batch_size=32,
    snapshot_epochs=SNAPSHOTS_AT
)

print(f"\n最終 Train error rate = {train_err_hist[-1]:.4f}")
print(f"最終 Test  error rate = {test_err_hist[-1]:.4f}")

# %% [Cell 7] (b) 畫學習曲線（loss + error rate）
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(loss_hist)
axes[0].set_title("Learning Curve (Cross-Entropy Loss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].plot(train_err_hist, label="Train error rate")
axes[1].plot(test_err_hist, label="Test error rate")
axes[1].set_title("Error Rate")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("learning_curve_classification.png", dpi=150)
plt.show()

# %% [Cell 8] (c) 畫出不同訓練階段的 latent feature 分布（2D，直接畫）
fig, axes = plt.subplots(1, len(SNAPSHOTS_AT), figsize=(4 * len(SNAPSHOTS_AT), 4))
for i, ep in enumerate(SNAPSHOTS_AT):
    feat = latent_snap[ep]
    mask_b = y_train_idx == 0
    mask_g = y_train_idx == 1
    axes[i].scatter(feat[mask_b, 0], feat[mask_b, 1], c="navy", s=15, label="Class b")
    axes[i].scatter(feat[mask_g, 0], feat[mask_g, 1], c="green", s=15, label="Class g")
    axes[i].set_title(f"2D feature @ epoch {ep}")
    axes[i].legend()

plt.tight_layout()
plt.savefig("latent_distribution_2d.png", dpi=150)
plt.show()

# %% [Cell 9] (c) 比較不同的 latent_dim（輸出層前一層的節點數）
# 當 latent_dim > 2 時，用手刻 PCA（只用 numpy SVD，不是 ML 套件）投影到 2D 才能畫圖
def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

LATENT_DIMS_TO_COMPARE = [2, 5, 10]
FINAL_EPOCH = EPOCHS

fig, axes = plt.subplots(1, len(LATENT_DIMS_TO_COMPARE), figsize=(4 * len(LATENT_DIMS_TO_COMPARE), 4))

for i, dim in enumerate(LATENT_DIMS_TO_COMPARE):
    m, lh, trh, teh, snaps = train_model(
        X_train, Y_train, y_train_idx, X_test, Y_test, y_test_idx,
        layer_sizes=[34, 32, 16, dim, 2],
        lr=0.1, epochs=FINAL_EPOCH, batch_size=32,
        snapshot_epochs=[FINAL_EPOCH]
    )
    feat = snaps[FINAL_EPOCH]
    feat_2d = feat if dim == 2 else pca_2d(feat)

    mask_b = y_train_idx == 0
    mask_g = y_train_idx == 1
    axes[i].scatter(feat_2d[mask_b, 0], feat_2d[mask_b, 1], c="navy", s=15, label="Class b")
    axes[i].scatter(feat_2d[mask_g, 0], feat_2d[mask_g, 1], c="green", s=15, label="Class g")
    axes[i].set_title(f"latent_dim={dim} (test_err={teh[-1]:.3f})")
    axes[i].legend()

plt.tight_layout()
plt.savefig("compare_latent_dims.png", dpi=150)
plt.show()

# %% [Cell 10] 結果總表（放進報告用）
print("===== 結果 =====")
print(f"Train error rate = {train_err_hist[-1]:.4f}")
print(f"Test  error rate = {test_err_hist[-1]:.4f}")
