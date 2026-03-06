"""
experiments.py  –  Question 4 experiments
Runs all experiments for the UWaveGestureLibrary CNN and RNN models.
Produces plots and prints a summary table.

Usage:
    python experiments.py
    (edit TRAIN_PATH / TEST_PATH if needed)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from assignment3 import UWaveGestureLibraryDataset

# ── paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "UWaveGestureLibrary_TRAIN.csv"
TEST_PATH  = "UWaveGestureLibrary_TEST.csv"
OUT_DIR    = "."          # directory where plots are saved


# ══════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════

class CNNModel(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        # Compute flattened size: after two Conv1d+MaxPool1d blocks
        # 315 → (315 - k + 1) // 2 → (... - k + 1) // 2
        L1 = (315 - kernel_size + 1) // 2
        L2 = (L1  - kernel_size + 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(3,  16, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * L2, 64), nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class RNNModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)       # (batch, 315, 3)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════

def get_loaders(filepath, val_frac=0.2, batch_size=32, seed=42):
    dataset   = UWaveGestureLibraryDataset(filepath)
    n_val     = max(1, int(val_frac * len(dataset)))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            labels = torch.argmax(y, dim=1)
            preds  = torch.argmax(model(x), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train for `epochs` epochs; return per-epoch (train_acc, val_acc)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_accs, val_accs = [], []

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            labels = torch.argmax(y, dim=1)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()
        train_accs.append(accuracy(model, train_loader))
        val_accs.append(accuracy(model, val_loader))

    return train_accs, val_accs


# ══════════════════════════════════════════════════════════════════════════
# Experiment helpers
# ══════════════════════════════════════════════════════════════════════════

def save_plot(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# (a) CNN kernel size experiment
# ══════════════════════════════════════════════════════════════════════════

def experiment_a(train_path, fixed_epochs=20):
    kernel_sizes = [3, 5, 7, 11, 15]
    print(f"\n=== (a) CNN kernel size  (epochs={fixed_epochs}) ===")
    results = {}   # kernel_size → (final_train_acc, final_val_acc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"(a) CNN – kernel size experiment  (epochs={fixed_epochs})", fontsize=13)

    train_loader, val_loader = get_loaders(train_path)

    for k in kernel_sizes:
        model = CNNModel(kernel_size=k)
        tr, va = train_model(model, train_loader, val_loader, epochs=fixed_epochs)
        results[k] = (tr[-1], va[-1])
        xs = range(1, fixed_epochs + 1)
        axes[0].plot(xs, tr, marker='o', markersize=3, label=f"k={k}")
        axes[1].plot(xs, va, marker='o', markersize=3, label=f"k={k}")
        print(f"  kernel={k:2d}  train={tr[-1]:.4f}  val={va[-1]:.4f}")

    for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.4)

    save_plot(fig, "exp_a_cnn_kernel_size.png")
    return results


# ══════════════════════════════════════════════════════════════════════════
# (b) RNN hidden size experiment
# ══════════════════════════════════════════════════════════════════════════

def experiment_b(train_path, fixed_epochs=20):
    hidden_sizes = [16, 32, 64, 128, 256]
    print(f"\n=== (b) RNN hidden size  (epochs={fixed_epochs}) ===")
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"(b) RNN – hidden size experiment  (epochs={fixed_epochs})", fontsize=13)

    train_loader, val_loader = get_loaders(train_path)

    for h in hidden_sizes:
        model = RNNModel(hidden_size=h)
        tr, va = train_model(model, train_loader, val_loader, epochs=fixed_epochs)
        results[h] = (tr[-1], va[-1])
        xs = range(1, fixed_epochs + 1)
        axes[0].plot(xs, tr, marker='o', markersize=3, label=f"h={h}")
        axes[1].plot(xs, va, marker='o', markersize=3, label=f"h={h}")
        print(f"  hidden={h:3d}  train={tr[-1]:.4f}  val={va[-1]:.4f}")

    for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.4)

    save_plot(fig, "exp_b_rnn_hidden_size.png")
    return results


# ══════════════════════════════════════════════════════════════════════════
# (c) Epoch experiment – both CNN and RNN
# ══════════════════════════════════════════════════════════════════════════

def experiment_c(train_path, max_epochs=60,
                 best_cnn_kernel=5, best_rnn_hidden=64):
    print(f"\n=== (c) Epoch experiment  (max_epochs={max_epochs}) ===")

    train_loader, val_loader = get_loaders(train_path)

    cnn_model = CNNModel(kernel_size=best_cnn_kernel)
    rnn_model = RNNModel(hidden_size=best_rnn_hidden)

    cnn_tr, cnn_va = train_model(cnn_model, train_loader, val_loader, epochs=max_epochs)
    rnn_tr, rnn_va = train_model(rnn_model, train_loader, val_loader, epochs=max_epochs)

    xs = range(1, max_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"(c) Epochs experiment  (CNN k={best_cnn_kernel}, RNN h={best_rnn_hidden})", fontsize=13)

    for ax, tr, va, name in zip(axes,
                                 [cnn_tr, rnn_tr], [cnn_va, rnn_va],
                                 ["CNN", "RNN"]):
        ax.plot(xs, tr, label="Train", marker='o', markersize=2)
        ax.plot(xs, va, label="Validation", marker='s', markersize=2)
        ax.set_title(f"{name} Accuracy vs Epoch")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend(); ax.grid(True, alpha=0.4)

    save_plot(fig, "exp_c_epochs.png")

    print(f"  CNN  train={cnn_tr[-1]:.4f}  val={cnn_va[-1]:.4f}")
    print(f"  RNN  train={rnn_tr[-1]:.4f}  val={rnn_va[-1]:.4f}")

    return cnn_model, rnn_model, cnn_tr[-1], cnn_va[-1], rnn_tr[-1], rnn_va[-1]


# ══════════════════════════════════════════════════════════════════════════
# (d) Test-set evaluation
# ══════════════════════════════════════════════════════════════════════════

def experiment_d(cnn_model, rnn_model, test_path):
    print(f"\n=== (d) Test-set performance ===")
    test_ds     = UWaveGestureLibraryDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    cnn_test = accuracy(cnn_model, test_loader)
    rnn_test = accuracy(rnn_model, test_loader)

    print(f"  CNN  test accuracy: {cnn_test:.4f}")
    print(f"  RNN  test accuracy: {rnn_test:.4f}")
    return cnn_test, rnn_test


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── (a) CNN kernel size ────────────────────────────────────────────────
    kernel_results = experiment_a(TRAIN_PATH, fixed_epochs=20)
    best_kernel = max(kernel_results, key=lambda k: kernel_results[k][1])
    print(f"\n  → Best kernel size: {best_kernel}  "
          f"(val acc={kernel_results[best_kernel][1]:.4f})")

    # ── (b) RNN hidden size ────────────────────────────────────────────────
    hidden_results = experiment_b(TRAIN_PATH, fixed_epochs=20)
    best_hidden = max(hidden_results, key=lambda h: hidden_results[h][1])
    print(f"\n  → Best hidden size: {best_hidden}  "
          f"(val acc={hidden_results[best_hidden][1]:.4f})")

    # ── (c) Epoch experiment with best params ──────────────────────────────
    (best_cnn, best_rnn,
     cnn_train_acc, cnn_val_acc,
     rnn_train_acc, rnn_val_acc) = experiment_c(
        TRAIN_PATH,
        max_epochs=60,
        best_cnn_kernel=best_kernel,
        best_rnn_hidden=best_hidden,
    )

    # ── (d) Test performance ───────────────────────────────────────────────
    cnn_test_acc, rnn_test_acc = experiment_d(best_cnn, best_rnn, TEST_PATH)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("SUMMARY")
    print("=" * 58)
    print(f"{'Model':<8} {'Param':<20} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 58)
    print(f"{'CNN':<8} {'kernel=' + str(best_kernel):<20} "
          f"{cnn_train_acc:>7.4f} {cnn_val_acc:>7.4f} {cnn_test_acc:>7.4f}")
    print(f"{'RNN':<8} {'hidden=' + str(best_hidden):<20} "
          f"{rnn_train_acc:>7.4f} {rnn_val_acc:>7.4f} {rnn_test_acc:>7.4f}")
    print("=" * 58)

    print("\nDone. Plots saved:")
    print("  exp_a_cnn_kernel_size.png")
    print("  exp_b_rnn_hidden_size.png")
    print("  exp_c_epochs.png")