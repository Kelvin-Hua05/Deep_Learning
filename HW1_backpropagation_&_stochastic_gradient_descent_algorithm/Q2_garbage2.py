# %% [Cell 1] Imports & Basic Settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

DATA_PATH = "2026_ionosphere_data.csv"

# Training settings
EPOCHS = 1000
BATCH_SIZE = 32

# Initial learning rate
LR = 0.05

# Learning-rate decay
LR_DECAY = 0.995

# L2 regularization strength
L2_LAMBDA = 1e-4

# Latent dimensions for part (c)
LATENT_DIMS_TO_COMPARE = [2, 5, 10]

# Training stages for latent visualization
SNAPSHOTS_AT = [10, 100, 500, 1000]


# %% [Cell 2] Load Ionosphere Dataset

# The CSV file has no header
df = pd.read_csv(DATA_PATH, header=None)

print("Data shape:", df.shape)

# Expected:
# 351 samples
# 34 features + 1 label column
assert df.shape[1] == 35, "Expected 35 columns."

X_raw = df.iloc[:, :-1].to_numpy(dtype=np.float64)
labels = df.iloc[:, -1].to_numpy()

print("Number of samples:", X_raw.shape[0])
print("Number of features:", X_raw.shape[1])

# ---------------------------------------------------------
# Label encoding
#
# b -> 0
# g -> 1
# ---------------------------------------------------------

y_idx = (labels == "g").astype(int)

# One-hot encoding
Y_onehot = np.eye(2)[y_idx]

print("Class b:", np.sum(y_idx == 0))
print("Class g:", np.sum(y_idx == 1))


# %% [Cell 3] Stratified 80/20 Train-Test Split

def stratified_split(X, Y, y, train_ratio=0.8, seed=42):

    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    classes = np.unique(y)

    for c in classes:

        class_indices = np.where(y == c)[0]

        rng.shuffle(class_indices)

        n_train_class = int(len(class_indices) * train_ratio)

        train_indices.extend(class_indices[:n_train_class])
        test_indices.extend(class_indices[n_train_class:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Shuffle final train/test sets
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        Y[train_indices],
        Y[test_indices],
        y[train_indices],
        y[test_indices]
    )


X_train_raw, X_test_raw, \
Y_train, Y_test, \
y_train_idx, y_test_idx = stratified_split(
    X_raw,
    Y_onehot,
    y_idx,
    train_ratio=0.8,
    seed=42
)

print("\nTrain size:", X_train_raw.shape)
print("Test size :", X_test_raw.shape)

print("\nTraining class distribution:")
print("b =", np.sum(y_train_idx == 0))
print("g =", np.sum(y_train_idx == 1))

print("\nTesting class distribution:")
print("b =", np.sum(y_test_idx == 0))
print("g =", np.sum(y_test_idx == 1))


# %% [Cell 4] Feature Standardization
#
# IMPORTANT:
# Mean/std are calculated ONLY from training data.
# This avoids data leakage from the test set.

X_mean = X_train_raw.mean(axis=0)
X_std = X_train_raw.std(axis=0)

# Avoid division by zero
X_std[X_std == 0] = 1.0

X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std

print("\nNormalization completed.")
print("Training mean approximately:", np.mean(X_train))
print("Training std approximately :", np.std(X_train))


# %% [Cell 5] MLP Neural Network
#
# Architecture:
#
# 34
#  ↓
# 32 ReLU
#  ↓
# 16 ReLU
#  ↓
# latent_dim ReLU
#  ↓
# 2 Softmax
#
# The latent_dim is the number of nodes
# in the layer before the output layer.

class MLP:

    def __init__(
        self,
        layer_sizes,
        lr=0.05,
        lr_decay=0.995,
        l2_lambda=1e-4
    ):

        self.layer_sizes = layer_sizes
        self.lr_initial = lr
        self.lr = lr
        self.lr_decay = lr_decay
        self.l2_lambda = l2_lambda

        self.L = len(layer_sizes) - 1

        self.W = []
        self.b = []

        # He initialization for ReLU
        for i in range(self.L):

            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            W = (
                np.random.randn(fan_in, fan_out)
                * np.sqrt(2.0 / fan_in)
            )

            b = np.zeros((1, fan_out))

            self.W.append(W)
            self.b.append(b)

    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    @staticmethod
    def relu(z):

        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):

        return (z > 0).astype(float)

    # -----------------------------------------------------
    # Softmax
    # -----------------------------------------------------

    @staticmethod
    def softmax(z):

        # Numerical stability
        z = z - np.max(
            z,
            axis=1,
            keepdims=True
        )

        exp_z = np.exp(z)

        return exp_z / np.sum(
            exp_z,
            axis=1,
            keepdims=True
        )

    # -----------------------------------------------------
    # Forward propagation
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Backward propagation
    # -----------------------------------------------------

    def backward(self, Y):

        m = Y.shape[0]

        y_pred = self.a_cache[-1]

        # Softmax + cross entropy derivative
        dZ = (y_pred - Y) / m

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        for i in reversed(range(self.L)):

            A_prev = self.a_cache[i]

            # Cross entropy gradient
            grads_W[i] = A_prev.T @ dZ

            grads_b[i] = np.sum(
                dZ,
                axis=0,
                keepdims=True
            )

            # L2 regularization
            grads_W[i] += self.l2_lambda * self.W[i]

            if i > 0:

                dA_prev = dZ @ self.W[i].T

                dZ = (
                    dA_prev
                    * self.relu_grad(
                        self.z_cache[i - 1]
                    )
                )

        # Update parameters
        for i in range(self.L):

            self.W[i] -= (
                self.lr * grads_W[i]
            )

            self.b[i] -= (
                self.lr * grads_b[i]
            )

    # -----------------------------------------------------
    # Prediction probability
    # -----------------------------------------------------

    def predict_proba(self, X):

        return self.forward(X)

    # -----------------------------------------------------
    # Prediction class
    # -----------------------------------------------------

    def predict(self, X):

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)

    # -----------------------------------------------------
    # Latent feature
    #
    # Output of the layer BEFORE final output layer
    # -----------------------------------------------------

    def latent(self, X):

        self.forward(X)

        return self.a_cache[-2]

    # -----------------------------------------------------
    # Learning rate decay
    # -----------------------------------------------------

    def decay_learning_rate(self):

        self.lr *= self.lr_decay


# %% [Cell 6] Loss & Error Functions

def cross_entropy(
    y_pred,
    Y_onehot,
    model=None,
    eps=1e-12
):

    # Avoid log(0)
    y_pred = np.clip(
        y_pred,
        eps,
        1.0
    )

    ce = -np.mean(
        np.sum(
            Y_onehot * np.log(y_pred),
            axis=1
        )
    )

    # Optional L2 regularization
    if model is not None:

        reg = 0.0

        for W in model.W:

            reg += np.sum(W ** 2)

        ce += (
            model.l2_lambda
            / 2
            * reg
        )

    return ce


def error_rate(
    y_pred,
    y_true_idx
):

    pred_idx = np.argmax(
        y_pred,
        axis=1
    )

    return np.mean(
        pred_idx != y_true_idx
    )


def accuracy(
    y_pred,
    y_true_idx
):

    return 1.0 - error_rate(
        y_pred,
        y_true_idx
    )


# %% [Cell 7] Training Function

def train_model(
    X_train,
    Y_train,
    y_train_idx,
    X_test,
    Y_test,
    y_test_idx,
    layer_sizes,
    lr,
    lr_decay,
    l2_lambda,
    epochs,
    batch_size,
    snapshot_epochs
):

    model = MLP(
        layer_sizes=layer_sizes,
        lr=lr,
        lr_decay=lr_decay,
        l2_lambda=l2_lambda
    )

    n = X_train.shape[0]

    loss_history = []

    train_err_history = []
    test_err_history = []

    train_acc_history = []
    test_acc_history = []

    latent_snapshots = {}

    for epoch in range(1, epochs + 1):

        # Shuffle training data
        perm = np.random.permutation(n)

        Xs = X_train[perm]
        Ys = Y_train[perm]

        # Mini-batch training
        for start in range(
            0,
            n,
            batch_size
        ):

            xb = Xs[
                start:start + batch_size
            ]

            yb = Ys[
                start:start + batch_size
            ]

            # Forward
            pred = model.forward(xb)

            # Backward + update
            model.backward(yb)

        # -----------------------------------------------
        # Calculate full training loss
        # -----------------------------------------------

        train_pred = model.predict_proba(
            X_train
        )

        test_pred = model.predict_proba(
            X_test
        )

        train_loss = cross_entropy(
            train_pred,
            Y_train,
            model
        )

        test_loss = cross_entropy(
            test_pred,
            Y_test,
            model
        )

        # Use training loss as learning curve
        loss_history.append(
            train_loss
        )

        # Error rates
        train_err = error_rate(
            train_pred,
            y_train_idx
        )

        test_err = error_rate(
            test_pred,
            y_test_idx
        )

        train_err_history.append(
            train_err
        )

        test_err_history.append(
            test_err
        )

        # Accuracy
        train_acc_history.append(
            1.0 - train_err
        )

        test_acc_history.append(
            1.0 - test_err
        )

        # -----------------------------------------------
        # Save latent feature
        # -----------------------------------------------

        if epoch in snapshot_epochs:

            latent_snapshots[epoch] = (
                model.latent(X_train).copy()
            )

            print(
                f"Epoch {epoch:4d} | "
                f"Loss = {train_loss:.5f} | "
                f"Train Error = {train_err:.4f} | "
                f"Test Error = {test_err:.4f}"
            )

        # Learning rate decay
        model.decay_learning_rate()

    return (
        model,
        loss_history,
        train_err_history,
        test_err_history,
        train_acc_history,
        test_acc_history,
        latent_snapshots
    )


# %% [Cell 8] Main Model
#
# Main architecture:
#
# 34 → 32 → 16 → 2 → 2

MAIN_LATENT_DIM = 2

main_architecture = [
    34,
    32,
    16,
    MAIN_LATENT_DIM,
    2
]

print("\n===================================")
print("MAIN NETWORK ARCHITECTURE")
print("===================================")

print(
    f"Input layer       : {main_architecture[0]}"
)

for i in range(
    1,
    len(main_architecture) - 1
):

    print(
        f"Hidden layer {i}    : "
        f"{main_architecture[i]} ReLU"
    )

print(
    f"Output layer      : "
    f"{main_architecture[-1]} Softmax"
)

print("Loss              : Cross-Entropy")
print("Optimizer         : Mini-batch SGD")
print("Initial LR        :", LR)
print("LR decay          :", LR_DECAY)
print("L2 lambda         :", L2_LAMBDA)
print("Batch size        :", BATCH_SIZE)
print("Epochs            :", EPOCHS)


# %% [Cell 9] Train Main Model

(
    model,
    loss_hist,
    train_err_hist,
    test_err_hist,
    train_acc_hist,
    test_acc_hist,
    latent_snap
) = train_model(
    X_train,
    Y_train,
    y_train_idx,
    X_test,
    Y_test,
    y_test_idx,

    layer_sizes=main_architecture,

    lr=LR,
    lr_decay=LR_DECAY,
    l2_lambda=L2_LAMBDA,

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    snapshot_epochs=SNAPSHOTS_AT
)


# %% [Cell 10] Final Main Model Results

final_train_error = train_err_hist[-1]
final_test_error = test_err_hist[-1]

final_train_accuracy = train_acc_hist[-1]
final_test_accuracy = test_acc_hist[-1]

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(
    f"Training Error Rate : "
    f"{final_train_error:.4f} "
    f"({final_train_error * 100:.2f}%)"
)

print(
    f"Testing Error Rate  : "
    f"{final_test_error:.4f} "
    f"({final_test_error * 100:.2f}%)"
)

print(
    f"Training Accuracy   : "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy    : "
    f"{final_test_accuracy * 100:.2f}%"
)


# %% [Cell 11] Confusion Matrix

def confusion_matrix_binary(
    y_true,
    y_pred
):

    cm = np.zeros(
        (2, 2),
        dtype=int
    )

    for true, pred in zip(
        y_true,
        y_pred
    ):

        cm[true, pred] += 1

    return cm


test_predictions = model.predict(
    X_test
)

cm = confusion_matrix_binary(
    y_test_idx,
    test_predictions
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(
    "             Pred b    Pred g"
)

print(
    f"True b      {cm[0,0]:6d}    {cm[0,1]:6d}"
)

print(
    f"True g      {cm[1,0]:6d}    {cm[1,1]:6d}"
)


# %% [Cell 12] (b) Learning Curve
#
# Plot:
# 1. Cross-Entropy loss
# 2. Train/Test error rate

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

# -----------------------------------------------
# Cross entropy
# -----------------------------------------------

axes[0].plot(
    range(1, EPOCHS + 1),
    loss_hist
)

axes[0].set_title(
    "Learning Curve - Cross Entropy"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Loss"
)

axes[0].grid(
    alpha=0.3
)

# -----------------------------------------------
# Error rate
# -----------------------------------------------

axes[1].plot(
    range(1, EPOCHS + 1),
    train_err_hist,
    label="Train error"
)

axes[1].plot(
    range(1, EPOCHS + 1),
    test_err_hist,
    label="Test error"
)

axes[1].set_title(
    "Training / Testing Error Rate"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Error Rate"
)

axes[1].legend()

axes[1].grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "Q2_learning_curve.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# %% [Cell 13] (c) PCA Function
#
# PCA is ONLY used to visualize latent features
# when latent dimension > 2.
#
# The classifier itself still uses the original
# latent features.

def pca_2d(X):

    X_centered = (
        X - X.mean(
            axis=0,
            keepdims=True
        )
    )

    U, S, Vt = np.linalg.svd(
        X_centered,
        full_matrices=False
    )

    return (
        X_centered
        @ Vt[:2].T
    )


# %% [Cell 14] (c) Latent Distribution
#
# Main model:
# latent_dim = 2
#
# Compare latent representations
# at different training stages.

mask_b = (
    y_train_idx == 0
)

mask_g = (
    y_train_idx == 1
)

fig, axes = plt.subplots(
    1,
    len(SNAPSHOTS_AT),
    figsize=(16, 4)
)

for i, epoch in enumerate(
    SNAPSHOTS_AT
):

    feat = latent_snap[epoch]

    axes[i].scatter(
        feat[mask_b, 0],
        feat[mask_b, 1],
        s=18,
        label="Class b"
    )

    axes[i].scatter(
        feat[mask_g, 0],
        feat[mask_g, 1],
        s=18,
        label="Class g"
    )

    axes[i].set_title(
        f"Latent Features - Epoch {epoch}"
    )

    axes[i].set_xlabel(
        "Latent Feature 1"
    )

    axes[i].set_ylabel(
        "Latent Feature 2"
    )

    axes[i].legend()

    axes[i].grid(
        alpha=0.2
    )

plt.tight_layout()

plt.savefig(
    "Q2_latent_training_stages.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# %% [Cell 15] Compare Different Latent Dimensions
#
# latent_dim = 2, 5, 10
#
# Each model is trained independently.

comparison_results = []

latent_models = {}

for dim in LATENT_DIMS_TO_COMPARE:

    print("\n===================================")
    print(
        f"Training latent_dim = {dim}"
    )
    print("===================================")

    architecture = [
        34,
        32,
        16,
        dim,
        2
    ]

    (
        m,
        lh,
        trh,
        teh,
        tr_acc,
        te_acc,
        snaps
    ) = train_model(
        X_train,
        Y_train,
        y_train_idx,
        X_test,
        Y_test,
        y_test_idx,

        layer_sizes=architecture,

        lr=LR,
        lr_decay=LR_DECAY,
        l2_lambda=L2_LAMBDA,

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        snapshot_epochs=SNAPSHOTS_AT
    )

    latent_models[dim] = {
        "model": m,
        "loss": lh,
        "train_error": trh,
        "test_error": teh,
        "train_accuracy": tr_acc,
        "test_accuracy": te_acc,
        "snapshots": snaps
    }

    comparison_results.append({
        "Latent Dimension": dim,
        "Train Error": trh[-1],
        "Test Error": teh[-1],
        "Train Accuracy": tr_acc[-1],
        "Test Accuracy": te_acc[-1]
    })


# %% [Cell 16] Comparison Table

comparison_df = pd.DataFrame(
    comparison_results
)

print("\n===================================")
print("LATENT DIMENSION COMPARISON")
print("===================================")

print(
    comparison_df.to_string(
        index=False
    )
)


# %% [Cell 17] Plot Final Latent Distributions
#
# For latent_dim > 2:
# PCA -> 2D
#
# IMPORTANT:
# PCA is only for visualization.

fig, axes = plt.subplots(
    1,
    len(LATENT_DIMS_TO_COMPARE),
    figsize=(15, 4)
)

for i, dim in enumerate(
    LATENT_DIMS_TO_COMPARE
):

    feat = latent_models[
        dim
    ]["snapshots"][EPOCHS]

    if dim == 2:

        feat_2d = feat

    else:

        feat_2d = pca_2d(feat)

    axes[i].scatter(
        feat_2d[mask_b, 0],
        feat_2d[mask_b, 1],
        s=18,
        label="Class b"
    )

    axes[i].scatter(
        feat_2d[mask_g, 0],
        feat_2d[mask_g, 1],
        s=18,
        label="Class g"
    )

    test_error = latent_models[
        dim
    ]["test_error"][-1]

    axes[i].set_title(
        f"Latent Dim = {dim}\n"
        f"Test Error = {test_error:.3f}"
    )

    axes[i].set_xlabel(
        "Component 1"
    )

    axes[i].set_ylabel(
        "Component 2"
    )

    axes[i].legend()

    axes[i].grid(
        alpha=0.2
    )

plt.tight_layout()

plt.savefig(
    "Q2_compare_latent_dimensions.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# %% [Cell 18] (c) Training Stage × Latent Dimension
#
# This is the most complete visualization for part (c).
#
# Rows:
# latent dimension = 2, 5, 10
#
# Columns:
# epoch = 10, 100, 500, 1000

fig, axes = plt.subplots(
    len(LATENT_DIMS_TO_COMPARE),
    len(SNAPSHOTS_AT),
    figsize=(16, 11)
)

for row, dim in enumerate(
    LATENT_DIMS_TO_COMPARE
):

    snaps = latent_models[
        dim
    ]["snapshots"]

    for col, epoch in enumerate(
        SNAPSHOTS_AT
    ):

        feat = snaps[epoch]

        if dim == 2:

            feat_2d = feat

        else:

            feat_2d = pca_2d(feat)

        ax = axes[row, col]

        ax.scatter(
            feat_2d[mask_b, 0],
            feat_2d[mask_b, 1],
            s=12,
            label="b"
        )

        ax.scatter(
            feat_2d[mask_g, 0],
            feat_2d[mask_g, 1],
            s=12,
            label="g"
        )

        ax.set_title(
            f"Latent={dim}, Epoch={epoch}"
        )

        ax.grid(
            alpha=0.2
        )

        if col == 0:

            ax.set_ylabel(
                f"Latent {dim}\nComponent 2"
            )

        if row == len(
            LATENT_DIMS_TO_COMPARE
        ) - 1:

            ax.set_xlabel(
                "Component 1"
            )

        if row == 0 and col == 0:

            ax.legend()

plt.suptitle(
    "Latent Feature Distribution at Different Training Stages",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "Q2_latent_stage_dimension_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# %% [Cell 19] Compare Error Rates
#
# Bar chart for different latent dimensions.

dims = comparison_df[
    "Latent Dimension"
].to_numpy()

train_errors = comparison_df[
    "Train Error"
].to_numpy()

test_errors = comparison_df[
    "Test Error"
].to_numpy()

x = np.arange(
    len(dims)
)

width = 0.35

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    x - width / 2,
    train_errors,
    width,
    label="Train Error"
)

plt.bar(
    x + width / 2,
    test_errors,
    width,
    label="Test Error"
)

plt.xticks(
    x,
    dims
)

plt.xlabel(
    "Latent Dimension"
)

plt.ylabel(
    "Error Rate"
)

plt.title(
    "Effect of Latent Dimension on Classification Error"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "Q2_latent_dimension_error_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# %% [Cell 20] Final Report Summary

print("\n")
print("====================================================")
print("                  Q2 FINAL SUMMARY")
print("====================================================")

print("\n(a) Classification:")
print("----------------------------------------")
print("Loss function : Cross-Entropy")
print("Optimizer     : Mini-batch SGD")

print("\n(b) Network:")
print("----------------------------------------")
print("Architecture  : 34 -> 32 -> 16 -> 2 -> 2")
print("Activation    : ReLU + Softmax")

print("\nFinal Performance:")
print("----------------------------------------")

print(
    f"Training Accuracy : "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"Training Error    : "
    f"{final_train_error * 100:.2f}%"
)

print(
    f"Testing Accuracy  : "
    f"{final_test_accuracy * 100:.2f}%"
)

print(
    f"Testing Error     : "
    f"{final_test_error * 100:.2f}%"
)

print("\n(c) Latent Dimension Comparison:")
print("----------------------------------------")

print(
    comparison_df.to_string(
        index=False
    )
)

print("\nGenerated figures:")
print("----------------------------------------")
print("1. Q2_learning_curve.png")
print("2. Q2_latent_training_stages.png")
print("3. Q2_compare_latent_dimensions.png")
print("4. Q2_latent_stage_dimension_comparison.png")
print("5. Q2_latent_dimension_error_comparison.png")

print("\n====================================================")
print("                     DONE")
print("====================================================")
