import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

np.random.seed(0)

IMG_PATH = "2026_NYCU.png"
IMG_SIZE = 80

if not os.path.exists(IMG_PATH):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((10, 10, 70, 70), outline=(30, 60, 200), width=4)
    draw.rectangle((25, 25, 55, 55), fill=(220, 60, 60))
    draw.line((0, 40, 80, 40), fill=(20, 150, 60), width=2)
    img.save(IMG_PATH)
    print(f"Not find {IMG_PATH}")

img = Image.open(IMG_PATH).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
img_arr = np.asarray(img).astype(np.float64) / 255.0  # (H, W, 3), 值域 [0,1]
H, W, _ = img_arr.shape
print("圖片大小:", img_arr.shape)

ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
coords = np.stack([xs.ravel(), ys.ravel()], axis=1)      # (N, 2)
colors = img_arr.reshape(-1, 3)                           # (N, 3)
N = coords.shape[0]
print("樣本數 N =", N)

# 切成90%/10%
perm = np.random.permutation(N)
coords, colors = coords[perm], colors[perm]
n_train = int(N * 0.9)
X_train, Y_train = coords[:n_train], colors[:n_train]
X_test, Y_test = coords[n_train:], colors[n_train:]
print("Train:", X_train.shape, " Test:", X_test.shape)

def fourier_features(x, num_freq=6):
    """
    x: (N, 2) 原始座標
    回傳: (N, 2 + 2*2*num_freq) 編碼後特徵 [x, y, sin(2^i*pi*x), cos(2^i*pi*x), sin(2^i*pi*y), cos(2^i*pi*y), ...]
    """
    freqs = 2.0 ** np.arange(num_freq) * np.pi   # (num_freq,)
    feats = [x]
    for f in freqs:
        feats.append(np.sin(f * x))
        feats.append(np.cos(f * x))
    return np.concatenate(feats, axis=1)

class MLP:
    def __init__(self, layer_sizes, lr=0.01):
        """
        layer_sizes: list, 例如 [2, 64, 64, 64, 3]
        隱藏層使用 ReLU，輸出層使用 Sigmoid（因為 RGB 值域是 [0,1]）
        """
        self.lr = lr
        self.L = len(layer_sizes) - 1
        self.W, self.b = [], []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            # He initialization
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
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        a = X
        self.z_cache = []
        self.a_cache = [a]
        for i in range(self.L):
            z = a @ self.W[i] + self.b[i]
            if i < self.L - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)  # 輸出層
            self.z_cache.append(z)
            self.a_cache.append(a)
        return a

    def backward(self, Y):
        m = Y.shape[0]
        # 輸出層: MSE loss 對 sigmoid 輸出的梯度
        y_pred = self.a_cache[-1]
        dA = (y_pred - Y) * 2.0 / (m * Y.shape[1])   # dLoss/dY
        dZ = dA * y_pred * (1 - y_pred)               # sigmoid 導數

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

    def predict(self, X):
        return self.forward(X)

def train_model(X_train, Y_train, X_test, Y_test, layer_sizes, lr, epochs,
                 batch_size, snapshot_epochs, full_coords):
    model = MLP(layer_sizes, lr=lr)
    n = X_train.shape[0]
    loss_history = []
    snapshots = {}

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)
        Xs, Ys = X_train[perm], Y_train[perm]
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = Ys[start:start + batch_size]
            pred = model.forward(xb)
            epoch_loss += np.sum((pred - yb) ** 2)
            model.backward(yb)
        loss_history.append(epoch_loss / n)  # 近似 MSE

        if epoch in snapshot_epochs:
            recon = model.predict(full_coords)
            snapshots[epoch] = recon.copy()
            print(f"Epoch {epoch:4d} | train MSE ~ {loss_history[-1]:.5f}")

    return model, loss_history, snapshots

def rms_error(model, X, Y):
    pred = model.predict(X)
    return np.sqrt(np.mean((pred - Y) ** 2) * 1.0)  # E_RMS 公式 (1/3N * sum) 等價於 mean

EPOCHS = 800
SNAPSHOTS_AT = [100, 200, 400, 800]

model_raw, loss_raw, snap_raw = train_model(
    X_train, Y_train, X_test, Y_test,
    layer_sizes=[2, 128, 128, 128, 3],
    lr=0.05, epochs=EPOCHS, batch_size=256,
    snapshot_epochs=SNAPSHOTS_AT, full_coords=coords
)

train_rms_raw = rms_error(model_raw, X_train, Y_train)
test_rms_raw = rms_error(model_raw, X_test, Y_test)
print(f"[RAW] Train RMS = {train_rms_raw:.5f} | Test RMS = {test_rms_raw:.5f}")

plt.figure(figsize=(5, 4))
plt.plot(loss_raw)
plt.title("Learning Curve (MSE Loss) - Raw Coordinates")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("learning_curve_raw.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, len(SNAPSHOTS_AT) + 1, figsize=(4 * (len(SNAPSHOTS_AT) + 1), 4))
axes[0].imshow(img_arr)
axes[0].set_title("Ground Truth")
axes[0].axis("off")

for i, ep in enumerate(SNAPSHOTS_AT):
    recon_img = np.clip(snap_raw[ep], 0, 1).reshape(H, W, 3)
    axes[i + 1].imshow(recon_img)
    axes[i + 1].set_title(f"Epoch {ep}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig("reconstruction_raw.png", dpi=150)
plt.show()

NUM_FREQ = 6
X_train_enc = fourier_features(X_train, NUM_FREQ)
X_test_enc = fourier_features(X_test, NUM_FREQ)
full_coords_enc = fourier_features(coords, NUM_FREQ)
enc_dim = X_train_enc.shape[1]
print("編碼後維度:", enc_dim)

model_enc, loss_enc, snap_enc = train_model(
    X_train_enc, Y_train, X_test_enc, Y_test,
    layer_sizes=[enc_dim, 128, 128, 128, 3],
    lr=0.05, epochs=EPOCHS, batch_size=256,
    snapshot_epochs=SNAPSHOTS_AT, full_coords=full_coords_enc
)

train_rms_enc = rms_error(model_enc, X_train_enc, Y_train)
test_rms_enc = rms_error(model_enc, X_test_enc, Y_test)
print(f"[ENCODED] Train RMS = {train_rms_enc:.5f} | Test RMS = {test_rms_enc:.5f}")

final_ep = SNAPSHOTS_AT[-1]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_arr)
axes[0].set_title("Ground Truth")
axes[0].axis("off")

axes[1].imshow(np.clip(snap_raw[final_ep], 0, 1).reshape(H, W, 3))
axes[1].set_title(f"Raw Coord (Epoch {final_ep})")
axes[1].axis("off")

axes[2].imshow(np.clip(snap_enc[final_ep], 0, 1).reshape(H, W, 3))
axes[2].set_title(f"Fourier Encoded (Epoch {final_ep})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("compare_raw_vs_encoded.png", dpi=150)
plt.show()

print("===== 結果 =====")
print(f"[Raw]      Train RMS = {train_rms_raw:.5f} | Test RMS = {test_rms_raw:.5f}")
print(f"[Encoded]  Train RMS = {train_rms_enc:.5f} | Test RMS = {test_rms_enc:.5f}")
