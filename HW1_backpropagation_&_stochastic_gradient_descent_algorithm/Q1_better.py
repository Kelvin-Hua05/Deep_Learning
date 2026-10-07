
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 0. Basic settings
# ============================================================

np.random.seed(42)

IMG_PATH = "2026_NYCU.png"
IMG_SIZE = 100

EPOCHS = 800
BATCH_SIZE = 256
LEARNING_RATE = 0.005

NUM_FREQ = 6
HIDDEN_UNITS = 128
SNAPSHOT_EPOCHS = [100, 200, 400, 800]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Load image and preprocess data
# ============================================================

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(
        f"找不到圖片：{IMG_PATH}\n"
    )

img = Image.open(IMG_PATH).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))

# RGB values: [0, 255] -> [0, 1]
img_arr = np.asarray(img, dtype=np.float64) / 255.0

H, W, C = img_arr.shape
N = H * W

print("Image shape:", img_arr.shape)
print("Number of pixels:", N)


# ============================================================
# 2. Build normalized coordinate grid
# ============================================================

x = np.linspace(-1.0, 1.0, W)
y = np.linspace(-1.0, 1.0, H)

xx, yy = np.meshgrid(x, y, indexing="xy")

# Preserve the original raster order for reconstruction.
full_coords = np.stack(
    [xx.reshape(-1), yy.reshape(-1)],
    axis=1
)

# Each pixel has one RGB target.
colors = img_arr.reshape(-1, 3)

print("Coordinates shape:", full_coords.shape)
print("Colors shape:", colors.shape)


# ============================================================
# 3. Shuffle and split: 90% train, 10% test
# ============================================================

indices = np.random.permutation(N)
n_train = int(0.9 * N)

train_idx = indices[:n_train]
test_idx = indices[n_train:]

X_train = full_coords[train_idx]
Y_train = colors[train_idx]

X_test = full_coords[test_idx]
Y_test = colors[test_idx]

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)


# ============================================================
# 4. Fourier feature mapping
# ============================================================

def fourier_features(x, num_freq=6):
    """
    Input:
        x: (N, 2), normalized coordinates

    Output:
        (N, 2 + 4*num_freq)

    Features include raw coordinates and sine/cosine
    features for both x and y coordinates.
    """
    freqs = (2.0 ** np.arange(num_freq)) * np.pi

    features = [x]

    for freq in freqs:
        features.append(np.sin(freq * x))
        features.append(np.cos(freq * x))

    return np.concatenate(features, axis=1)


# ============================================================
# 5. Define MLP from scratch using NumPy
# ============================================================

class MLP:
    def __init__(self, layer_sizes, lr=0.01):
        self.lr = lr
        self.L = len(layer_sizes) - 1

        self.W = []
        self.b = []

        for i in range(self.L):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # He initialization
            weight = (
                np.random.randn(fan_in, fan_out)
                * np.sqrt(2.0 / fan_in)
            )

            bias = np.zeros((1, fan_out))

            self.W.append(weight)
            self.b.append(bias)

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float64)

    @staticmethod
    def sigmoid(z):
        # Clip for numerical stability.
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        self.z_cache = []
        self.a_cache = [X]

        a = X

        for i in range(self.L):
            z = a @ self.W[i] + self.b[i]

            if i < self.L - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)

            self.z_cache.append(z)
            self.a_cache.append(a)

        return a

    def backward(self, Y):
        """
        Backpropagation for mean squared error:

        MSE = mean((prediction - target) ** 2)

        The mean is over both samples and RGB channels.
        """
        m = Y.shape[0]
        output_dim = Y.shape[1]

        y_pred = self.a_cache[-1]

        # dMSE / dA_output
        dA = 2.0 * (y_pred - Y) / (m * output_dim)

        # Sigmoid derivative
        dZ = dA * y_pred * (1.0 - y_pred)

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        for i in reversed(range(self.L)):
            A_prev = self.a_cache[i]

            grads_W[i] = A_prev.T @ dZ
            grads_b[i] = np.sum(
                dZ, axis=0, keepdims=True
            )

            if i > 0:
                dA_prev = dZ @ self.W[i].T
                dZ = (
                    dA_prev
                    * self.relu_grad(self.z_cache[i - 1])
                )

        # SGD parameter update
        for i in range(self.L):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def predict(self, X):
        return self.forward(X)


# ============================================================
# 6. Evaluation metrics
# ============================================================

def calculate_mse(model, X, Y):
    prediction = model.predict(X)
    return float(np.mean((prediction - Y) ** 2))


def calculate_rms(model, X, Y):
    return float(np.sqrt(calculate_mse(model, X, Y)))


# ============================================================
# 7. Training loop: mini-batch SGD
# ============================================================

def train_model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    layer_sizes,
    lr,
    epochs,
    batch_size,
    snapshot_epochs,
    full_coords
):
    model = MLP(layer_sizes, lr=lr)

    n = X_train.shape[0]

    train_mse_history = []
    test_mse_history = []
    snapshots = {}

    for epoch in range(1, epochs + 1):

        # Shuffle training samples at each epoch.
        perm = np.random.permutation(n)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        # Mini-batch updates.
        for start in range(0, n, batch_size):
            end = start + batch_size

            xb = X_shuffled[start:end]
            yb = Y_shuffled[start:end]

            model.forward(xb)
            model.backward(yb)

        # Evaluate after all updates in this epoch.
        train_mse = calculate_mse(
            model, X_train, Y_train
        )
        test_mse = calculate_mse(
            model, X_test, Y_test
        )

        train_mse_history.append(train_mse)
        test_mse_history.append(test_mse)

        # Save reconstruction in original image order.
        if epoch in snapshot_epochs:
            reconstruction = model.predict(full_coords)
            snapshots[epoch] = reconstruction.copy()

        if (
            epoch == 1
            or epoch % 50 == 0
            or epoch in snapshot_epochs
        ):
            train_rms = np.sqrt(train_mse)
            test_rms = np.sqrt(test_mse)

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train MSE: {train_mse:.6f} | "
                f"Test MSE: {test_mse:.6f} | "
                f"Train RMS: {train_rms:.6f} | "
                f"Test RMS: {test_rms:.6f}"
            )

    return (
        model,
        train_mse_history,
        test_mse_history,
        snapshots
    )


# ============================================================
# 8. Experiment A: raw coordinates
# ============================================================

print("\n========== RAW COORDINATES ==========")

raw_layer_sizes = [
    2,
    HIDDEN_UNITS,
    HIDDEN_UNITS,
    HIDDEN_UNITS,
    3
]

model_raw, loss_raw, test_loss_raw, snap_raw = train_model(
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    layer_sizes=raw_layer_sizes,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    snapshot_epochs=SNAPSHOT_EPOCHS,
    full_coords=full_coords
)

train_rms_raw = calculate_rms(
    model_raw, X_train, Y_train
)
test_rms_raw = calculate_rms(
    model_raw, X_test, Y_test
)

print("\nRaw-coordinate final results")
print("Train RMS:", train_rms_raw)
print("Test RMS :", test_rms_raw)


# ============================================================
# 9. Experiment B: Fourier encoded coordinates
# ============================================================

print("\n========== FOURIER ENCODING ==========")

X_train_enc = fourier_features(X_train, NUM_FREQ)
X_test_enc = fourier_features(X_test, NUM_FREQ)
full_coords_enc = fourier_features(full_coords, NUM_FREQ)

enc_dim = X_train_enc.shape[1]

print("Fourier feature dimension:", enc_dim)

enc_layer_sizes = [
    enc_dim,
    HIDDEN_UNITS,
    HIDDEN_UNITS,
    HIDDEN_UNITS,
    3
]

model_enc, loss_enc, test_loss_enc, snap_enc = train_model(
    X_train=X_train_enc,
    Y_train=Y_train,
    X_test=X_test_enc,
    Y_test=Y_test,
    layer_sizes=enc_layer_sizes,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    snapshot_epochs=SNAPSHOT_EPOCHS,
    full_coords=full_coords_enc
)

train_rms_enc = calculate_rms(
    model_enc, X_train_enc, Y_train
)
test_rms_enc = calculate_rms(
    model_enc, X_test_enc, Y_test
)

print("\nFourier-encoded final results")
print("Train RMS:", train_rms_enc)
print("Test RMS :", test_rms_enc)


# ============================================================
# 10. Plot learning curves
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    loss_raw,
    label="Raw Train MSE"
)
plt.plot(
    range(1, EPOCHS + 1),
    test_loss_raw,
    label="Raw Test MSE"
)
plt.plot(
    range(1, EPOCHS + 1),
    loss_enc,
    label="Fourier Train MSE"
)
plt.plot(
    range(1, EPOCHS + 1),
    test_loss_enc,
    label="Fourier Test MSE"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Learning Curves: Raw vs Fourier")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "learning_curves.png"),
    dpi=200
)
plt.show()


# ============================================================
# 11. Plot ground truth and epoch snapshots
# ============================================================

fig, axes = plt.subplots(
    2,
    len(SNAPSHOT_EPOCHS) + 1,
    figsize=(16, 7)
)

axes[0, 0].imshow(img_arr)
axes[0, 0].set_title("Ground Truth")
axes[0, 0].axis("off")

axes[1, 0].imshow(img_arr)
axes[1, 0].set_title("Ground Truth")
axes[1, 0].axis("off")

for j, epoch in enumerate(SNAPSHOT_EPOCHS, start=1):
    raw_img = np.clip(
        snap_raw[epoch].reshape(H, W, 3), 0.0, 1.0
    )

    enc_img = np.clip(
        snap_enc[epoch].reshape(H, W, 3), 0.0, 1.0
    )

    axes[0, j].imshow(raw_img)
    axes[0, j].set_title(f"Raw Epoch {epoch}")
    axes[0, j].axis("off")

    axes[1, j].imshow(enc_img)
    axes[1, j].set_title(f"Fourier Epoch {epoch}")
    axes[1, j].axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "epoch_reconstructions.png"),
    dpi=200
)
plt.show()


# ============================================================
# 12. Final side-by-side comparison
# ============================================================

raw_final = np.clip(
    snap_raw[EPOCHS].reshape(H, W, 3), 0.0, 1.0
)

enc_final = np.clip(
    snap_enc[EPOCHS].reshape(H, W, 3), 0.0, 1.0
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_arr)
axes[0].set_title("Ground Truth")

axes[1].imshow(raw_final)
axes[1].set_title(f"Raw (RMS={test_rms_raw:.4f})")

axes[2].imshow(enc_final)
axes[2].set_title(f"Fourier (RMS={test_rms_enc:.4f})")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "raw_vs_fourier.png"),
    dpi=200
)
plt.show()


# ============================================================
# 13. Save numerical results
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR, "regression_results.csv"
)

with open(
    results_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "Experiment",
        "Input Dimension",
        "Hidden Layers",
        "Hidden Units",
        "Epochs",
        "Learning Rate",
        "Batch Size",
        "Train MSE",
        "Test MSE",
        "Train RMS",
        "Test RMS"
    ])

    writer.writerow([
        "Raw",
        2,
        3,
        HIDDEN_UNITS,
        EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        calculate_mse(model_raw, X_train, Y_train),
        calculate_mse(model_raw, X_test, Y_test),
        train_rms_raw,
        test_rms_raw
    ])

    writer.writerow([
        "Fourier",
        enc_dim,
        3,
        HIDDEN_UNITS,
        EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        calculate_mse(model_enc, X_train_enc, Y_train),
        calculate_mse(model_enc, X_test_enc, Y_test),
        train_rms_enc,
        test_rms_enc
    ])

print("\n========== FINAL SUMMARY ==========")
print(f"Raw      Train RMS: {train_rms_raw:.6f}")
print(f"Raw      Test RMS : {test_rms_raw:.6f}")
print(f"Fourier  Train RMS: {train_rms_enc:.6f}")
print(f"Fourier  Test RMS : {test_rms_enc:.6f}")

print("\nSaved output files in:", OUTPUT_DIR)
print("- learning_curves.png")
print("- epoch_reconstructions.png")
print("- raw_vs_fourier.png")
print("- regression_results.csv")
