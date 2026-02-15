"""
=============================================================================
  SECURE GATEWAY — Visualization & Analysis Suite (Phase 5)
=============================================================================

  Generates all visual evidence for the project video presentation.
  Reads saved training data, embeddings, model, and produces 6 high-quality
  plots that demonstrate the system works mathematically.

  Visualizations:
      1. t-SNE Embedding Clusters    — "The system can distinguish people"
      2. Training Curves              — "The model learned effectively"
      3. Confusion Matrix Heatmap     — "The model makes correct decisions"
      4. Similarity Distributions     — "The threshold is data-driven"
      5. Per-Class Performance Bars   — "Every user is well-protected"
      6. System Architecture Diagram  — "How it all fits together"
      + Combined Dashboard            — All 6 in one figure

  Output:
      evaluation/figures/*.png (300 DPI)

  Usage:
      python evaluation/visualizer.py

  Dependencies:
      pip install matplotlib scikit-learn numpy torch

=============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Use non-interactive backend (works headless, no display required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ============================================================
#  PATH SETUP
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
#  CONFIGURATION
# ============================================================
try:
    from utils.config import (
        PROJECT_ROOT as _PROJECT_ROOT,
        PATHS, CLASSES, THRESHOLDS, TRAINING, EMBEDDINGS, DEVICE
    )
    print("[CONFIG] Loaded configuration from utils/config.py")

    FACE_EMBEDDINGS_PATH = PATHS.FACE_EMBEDDINGS
    VOICE_EMBEDDINGS_PATH = PATHS.VOICE_EMBEDDINGS
    FUSION_MODEL_PATH = PATHS.FUSION_MODEL
    USER_PROFILES_PATH = PATHS.USER_PROFILES
    DATA_SPLITS_PATH = PATHS.DATA_SPLITS
    TRAINING_HISTORY_PATH = os.path.join(PATHS.MODELS_DIR, "training_history.pt")
    MODELS_DIR = PATHS.MODELS_DIR

    CLASS_LABELS = CLASSES.ALL_CLASSES
    AUTHORIZED_USERS = CLASSES.AUTHORIZED_USERS
    NUM_CLASSES = CLASSES.NUM_CLASSES
    CLASS_TO_IDX = CLASSES.CLASS_TO_IDX

    CONFIDENCE_HIGH = THRESHOLDS.HIGH_CONFIDENCE
    CONFIDENCE_LOW = THRESHOLDS.LOW_CONFIDENCE

    FUSED_DIM = EMBEDDINGS.FUSED_EMBEDDING_DIM

    COMPUTE_DEVICE = DEVICE.COMPUTE

except ImportError:
    print("[CONFIG] WARNING: Could not import utils/config.py — using defaults")
    import torch as _t
    COMPUTE_DEVICE = _t.device("cpu")
    FACE_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "face_embeddings.pt")
    VOICE_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "voice_embeddings.pt")
    FUSION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fusion_model.pt")
    USER_PROFILES_PATH = os.path.join(PROJECT_ROOT, "models", "user_profiles.pt")
    DATA_SPLITS_PATH = os.path.join(PROJECT_ROOT, "models", "data_splits.pt")
    TRAINING_HISTORY_PATH = os.path.join(PROJECT_ROOT, "models", "training_history.pt")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    CLASS_LABELS = ["david", "itzhak", "yossi", "unknown"]
    AUTHORIZED_USERS = ["david", "itzhak", "yossi"]
    NUM_CLASSES = 4
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_LABELS)}
    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_LOW = 0.50
    FUSED_DIM = 704

# Output directory
FIGURES_DIR = os.path.join(PROJECT_ROOT, "evaluation", "figures")

# ============================================================
#  VISUAL STYLE
# ============================================================
# Consistent color scheme across all plots.
# Each authorized user gets a distinctive color; unknown gets gray.

USER_COLORS = {
    "david":   "#2196F3",  # Blue
    "itzhak":  "#FF9800",  # Orange
    "yossi":   "#4CAF50",  # Green
    "unknown": "#9E9E9E",  # Gray
}

# Fallback for any extra users
EXTRA_COLORS = ["#E91E63", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]

def get_color(label):
    """Get the color for a class label."""
    if label in USER_COLORS:
        return USER_COLORS[label]
    idx = list(CLASS_TO_IDX.keys()).index(label) if label in CLASS_TO_IDX else 0
    return EXTRA_COLORS[idx % len(EXTRA_COLORS)]


# Global style settings
STYLE_CONFIG = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}

plt.rcParams.update(STYLE_CONFIG)


# ============================================================
#  DATA LOADING
# ============================================================

import torch

def load_embeddings():
    """Load face and voice embedding files."""
    print("\n  Loading embeddings...")

    face_data = None
    voice_data = None

    if os.path.exists(FACE_EMBEDDINGS_PATH):
        face_data = torch.load(FACE_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
        total_face = sum(v.shape[0] for v in face_data.values())
        print(f"    Face embeddings:  {total_face} vectors across {len(face_data)} people")
    else:
        print(f"    ⚠ Face embeddings not found: {FACE_EMBEDDINGS_PATH}")

    if os.path.exists(VOICE_EMBEDDINGS_PATH):
        voice_data = torch.load(VOICE_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
        total_voice = sum(v.shape[0] for v in voice_data.values())
        print(f"    Voice embeddings: {total_voice} vectors across {len(voice_data)} people")
    else:
        print(f"    ⚠ Voice embeddings not found: {VOICE_EMBEDDINGS_PATH}")

    return face_data, voice_data


def load_data_splits():
    """Load saved train/val/test splits."""
    if os.path.exists(DATA_SPLITS_PATH):
        data = torch.load(DATA_SPLITS_PATH, map_location="cpu", weights_only=False)
        print(f"    Data splits:      loaded (train/val/test)")
        return data
    else:
        print(f"    ⚠ Data splits not found: {DATA_SPLITS_PATH}")
        return None


def load_training_history():
    """Load training history (loss/accuracy per epoch)."""
    if os.path.exists(TRAINING_HISTORY_PATH):
        history = torch.load(TRAINING_HISTORY_PATH, map_location="cpu", weights_only=False)
        n_epochs = len(history.get("train_loss", history.get("train_losses", [])))
        print(f"    Training history: {n_epochs} epochs")
        return history
    else:
        print(f"    ⚠ Training history not found: {TRAINING_HISTORY_PATH}")
        return None


def load_fusion_model():
    """Load the trained fusion model for inference."""
    if not os.path.exists(FUSION_MODEL_PATH):
        print(f"    ⚠ Fusion model not found: {FUSION_MODEL_PATH}")
        return None

    import torch.nn as nn

    checkpoint = torch.load(FUSION_MODEL_PATH, map_location="cpu", weights_only=False)

    # Read architecture from saved config (self-contained checkpoint)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        cfg = checkpoint["model_config"]
        input_dim = cfg.get("input_dim", FUSED_DIM)
        hidden1 = cfg.get("hidden1", 256)
        hidden2 = cfg.get("hidden2", 128)
        num_classes = cfg.get("num_classes", NUM_CLASSES)
        dropout1 = cfg.get("dropout1", 0.3)
        dropout2 = cfg.get("dropout2", 0.2)
    else:
        input_dim, hidden1, hidden2 = FUSED_DIM, 256, 128
        num_classes, dropout1, dropout2 = NUM_CLASSES, 0.3, 0.2

    class FusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout1),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout2),
                nn.Linear(hidden2, num_classes)
            )

        def forward(self, x):
            return self.network(x)

    model = FusionMLP()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Print model info
    info_parts = [f"{input_dim}→{hidden1}→{hidden2}→{num_classes}"]
    if isinstance(checkpoint, dict):
        test_acc = checkpoint.get("test_acc", None)
        best_epoch = checkpoint.get("best_epoch", None)
        if test_acc is not None:
            info_parts.append(f"test_acc={test_acc:.1%}")
        if best_epoch is not None:
            info_parts.append(f"epoch={best_epoch}")

    print(f"    Fusion model:     loaded ({', '.join(info_parts)})")
    return model


def load_user_profiles():
    """Load enrollment profiles."""
    if os.path.exists(USER_PROFILES_PATH):
        profiles = torch.load(USER_PROFILES_PATH, map_location="cpu", weights_only=False)
        users = [k for k in profiles.keys() if k != "unknown"]
        print(f"    User profiles:    {len(users)} enrolled ({', '.join(users)})")
        return profiles
    else:
        print(f"    ⚠ User profiles not found: {USER_PROFILES_PATH}")
        return None


# ============================================================
#  HELPER: Build fused test vectors from splits
# ============================================================

def build_fused_test_data(splits_data, face_data, voice_data):
    """
    Build paired fused vectors for the test set, matching the training logic.

    For each authorized user: pairs face[i] with voice[j] from that user's
    test set (genuine pairs, labeled as the user).

    For the unknown class: cross-person pairs (face from person A, voice from
    person B), labeled as unknown.

    Returns:
        fused_vectors: Tensor [N, 704]
        labels: Tensor [N] (class indices)
        label_names: list of str (class name per sample)
    """
    all_fused = []
    all_labels = []
    all_names = []

    splits = splits_data["splits"]
    test_splits = splits["test"]

    # Genuine pairs for each authorized user
    for person in AUTHORIZED_USERS:
        if person not in test_splits:
            continue

        face_emb = test_splits[person]["face"]
        voice_emb = test_splits[person]["voice"]

        n_face = face_emb.shape[0]
        n_voice = voice_emb.shape[0]

        if n_face == 0 or n_voice == 0:
            continue

        # Pair each face with a cycling voice (same as training logic)
        for i in range(n_face):
            j = i % n_voice
            fused = torch.cat([face_emb[i], voice_emb[j]], dim=0)
            all_fused.append(fused)
            all_labels.append(CLASS_TO_IDX[person])
            all_names.append(person)

    # Unknown pairs: cross-person face[A] + voice[B]
    persons = [p for p in AUTHORIZED_USERS if p in test_splits]
    for i, person_a in enumerate(persons):
        for j, person_b in enumerate(persons):
            if i == j:
                continue
            face_a = test_splits[person_a]["face"]
            voice_b = test_splits[person_b]["voice"]

            n_pairs = min(face_a.shape[0], voice_b.shape[0])
            # Limit unknown pairs to keep class balance reasonable
            n_pairs = min(n_pairs, 50)

            for k in range(n_pairs):
                fused = torch.cat([face_a[k], voice_b[k % voice_b.shape[0]]], dim=0)
                all_fused.append(fused)
                all_labels.append(CLASS_TO_IDX["unknown"])
                all_names.append("unknown")

    if not all_fused:
        return None, None, None

    fused_vectors = torch.stack(all_fused)
    labels = torch.tensor(all_labels, dtype=torch.long)

    return fused_vectors, labels, all_names


def build_fused_from_raw(face_data, voice_data):
    """
    Fallback: build fused vectors directly from all embeddings (no splits).
    Used when data_splits.pt is not available.
    """
    all_fused = []
    all_labels = []
    all_names = []

    for person in AUTHORIZED_USERS:
        if person not in face_data or person not in voice_data:
            continue

        face_emb = face_data[person]
        voice_emb = voice_data[person]
        n_face = face_emb.shape[0]
        n_voice = voice_emb.shape[0]

        # Sample a subset to keep it manageable for t-SNE
        n_pairs = min(n_face, 150)

        for i in range(n_pairs):
            j = i % n_voice
            fused = torch.cat([face_emb[i], voice_emb[j]], dim=0)
            all_fused.append(fused)
            all_labels.append(CLASS_TO_IDX[person])
            all_names.append(person)

    # Unknown pairs
    persons = [p for p in AUTHORIZED_USERS if p in face_data and p in voice_data]
    for i, pa in enumerate(persons):
        for j, pb in enumerate(persons):
            if i == j:
                continue
            n_pairs = min(face_data[pa].shape[0], voice_data[pb].shape[0], 50)
            for k in range(n_pairs):
                fused = torch.cat([
                    face_data[pa][k],
                    voice_data[pb][k % voice_data[pb].shape[0]]
                ], dim=0)
                all_fused.append(fused)
                all_labels.append(CLASS_TO_IDX["unknown"])
                all_names.append("unknown")

    if not all_fused:
        return None, None, None

    return torch.stack(all_fused), torch.tensor(all_labels, dtype=torch.long), all_names


# ============================================================
#  PLOT 1: t-SNE EMBEDDING CLUSTERS
# ============================================================

def plot_tsne_clusters(fused_vectors, labels, label_names, profiles=None, save_path=None):
    """
    Reduce 704-dim fused embeddings to 2D with t-SNE and plot colored clusters.
    """
    print("\n  [1/6] Generating t-SNE cluster plot...")
    from sklearn.manifold import TSNE

    X = fused_vectors.numpy()
    n_samples = X.shape[0]

    # Adjust perplexity for small datasets
    perplexity = min(30, max(5, n_samples // 4))

    print(f"    Running t-SNE ({n_samples} samples, perplexity={perplexity})...")
    start = time.time()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        learning_rate="auto",
        init="pca"
    )
    coords = tsne.fit_transform(X)
    elapsed = time.time() - start
    print(f"    t-SNE completed in {elapsed:.1f}s")

    # Also project enrollment profiles if available
    profile_coords = {}
    if profiles is not None:
        for person in AUTHORIZED_USERS:
            if person in profiles:
                prof = profiles[person]
                # Get face and voice mean embeddings
                face_mean = prof.get("face_mean", prof.get("face", prof.get("face_embedding", None)))
                voice_mean = prof.get("voice_mean", prof.get("voice", prof.get("voice_embedding", None)))
                if face_mean is not None and voice_mean is not None:
                    fused_prof = torch.cat([face_mean.cpu(), voice_mean.cpu()], dim=0).numpy()
                    # Project using the same t-SNE is not ideal, so we find the
                    # nearest cluster center as an approximation
                    mask = np.array(label_names) == person
                    if mask.any():
                        profile_coords[person] = coords[mask].mean(axis=0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot each class
    unique_labels = sorted(set(label_names))
    for label in unique_labels:
        mask = np.array(label_names) == label
        color = get_color(label)
        alpha = 0.35 if label == "unknown" else 0.7
        size = 20 if label == "unknown" else 40
        marker = "x" if label == "unknown" else "o"

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=label.capitalize(),
            alpha=alpha, s=size, marker=marker,
            edgecolors="white" if label != "unknown" else "none",
            linewidths=0.3
        )

    # Plot enrollment profile centers as stars
    for person, coord in profile_coords.items():
        color = get_color(person)
        ax.scatter(
            coord[0], coord[1],
            c=color, s=300, marker="*",
            edgecolors="white", linewidths=1.5,
            zorder=10
        )

    ax.set_title("Fused Embedding Space (t-SNE Projection)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="upper right", framealpha=0.8, facecolor="#1a1a2e", edgecolor="#555")
    ax.grid(True, alpha=0.15)

    # Add annotation for stars
    if profile_coords:
        ax.annotate("★ = Enrollment Profile Center", xy=(0.02, 0.02),
                     xycoords="axes fraction", fontsize=9, alpha=0.7, style="italic")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    ✓ Saved: {save_path}")
    plt.close(fig)
    return True


# ============================================================
#  PLOT 2: TRAINING CURVES
# ============================================================

def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss + accuracy curves over epochs.
    """
    print("\n  [2/6] Generating training curves...")

    if history is None:
        print("    ⚠ Skipped — no training history available")
        return False

    # Handle different key naming conventions
    train_loss = history.get("train_loss", history.get("train_losses", []))
    val_loss = history.get("val_loss", history.get("val_losses", []))
    train_acc = history.get("train_acc", history.get("train_accs", history.get("train_accuracy", [])))
    val_acc = history.get("val_acc", history.get("val_accs", history.get("val_accuracy", [])))

    # Convert tensors to lists if needed
    if hasattr(train_loss, "tolist"):
        train_loss = train_loss.tolist() if hasattr(train_loss, "tolist") else list(train_loss)
    if hasattr(val_loss, "tolist"):
        val_loss = val_loss.tolist() if hasattr(val_loss, "tolist") else list(val_loss)
    if hasattr(train_acc, "tolist"):
        train_acc = train_acc.tolist() if hasattr(train_acc, "tolist") else list(train_acc)
    if hasattr(val_acc, "tolist"):
        val_acc = val_acc.tolist() if hasattr(val_acc, "tolist") else list(val_acc)

    # Make sure they're plain lists of floats
    train_loss = [float(x) for x in train_loss]
    val_loss = [float(x) for x in val_loss]
    train_acc = [float(x) for x in train_acc]
    val_acc = [float(x) for x in val_acc]

    n_epochs = len(train_loss)
    if n_epochs == 0:
        print("    ⚠ Skipped — training history is empty")
        return False

    epochs = range(1, n_epochs + 1)

    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(val_loss) + 1 if val_loss else 0
    best_val_loss = min(val_loss) if val_loss else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Loss curves ---
    ax1.plot(epochs, train_loss, color="#2196F3", linewidth=2, label="Train Loss", alpha=0.9)
    if val_loss:
        ax1.plot(epochs, val_loss, color="#FF9800", linewidth=2, label="Val Loss", alpha=0.9)
        ax1.axvline(x=best_epoch, color="#4CAF50", linestyle="--", alpha=0.6,
                     label=f"Best Epoch ({best_epoch})")
        ax1.scatter([best_epoch], [best_val_loss], c="#4CAF50", s=100, zorder=10,
                     edgecolors="white", linewidths=1.5)

    ax1.set_title("Loss Over Training", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(framealpha=0.8, facecolor="#1a1a2e", edgecolor="#555")
    ax1.grid(True, alpha=0.2)

    # --- Accuracy curves ---
    # Convert to percentages if they are in 0-1 range
    train_acc_pct = [a * 100 if a <= 1.0 else a for a in train_acc]
    val_acc_pct = [a * 100 if a <= 1.0 else a for a in val_acc]

    ax2.plot(epochs, train_acc_pct, color="#2196F3", linewidth=2, label="Train Accuracy", alpha=0.9)
    if val_acc_pct:
        ax2.plot(epochs, val_acc_pct, color="#FF9800", linewidth=2, label="Val Accuracy", alpha=0.9)
        best_val_acc = val_acc_pct[best_epoch - 1] if best_epoch <= len(val_acc_pct) else 0
        ax2.axvline(x=best_epoch, color="#4CAF50", linestyle="--", alpha=0.6,
                     label=f"Best Epoch ({best_epoch})")
        ax2.scatter([best_epoch], [best_val_acc], c="#4CAF50", s=100, zorder=10,
                     edgecolors="white", linewidths=1.5)

    ax2.set_title("Accuracy Over Training", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([0, 105])
    ax2.legend(framealpha=0.8, facecolor="#1a1a2e", edgecolor="#555")
    ax2.grid(True, alpha=0.2)

    plt.suptitle("Fusion Model Training Progress", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    ✓ Saved: {save_path}")
    plt.close(fig)
    return True


# ============================================================
#  PLOT 3: CONFUSION MATRIX HEATMAP
# ============================================================

def plot_confusion_matrix(fused_vectors, labels, model, save_path=None):
    """
    Run inference on test data and plot a confusion matrix heatmap.
    """
    print("\n  [3/6] Generating confusion matrix...")

    if model is None or fused_vectors is None:
        print("    ⚠ Skipped — model or test data not available")
        return False

    from sklearn.metrics import confusion_matrix, accuracy_score

    # Run inference
    with torch.no_grad():
        logits = model(fused_vectors)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    y_true = labels.numpy()
    y_pred = preds.numpy()

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"    Test accuracy: {overall_acc:.1%}")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Normalize for display (percentages per row)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_pct = cm_norm / row_sums * 100

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Custom colormap: dark blue → bright green
    cmap = LinearSegmentedColormap.from_list("custom", ["#16213e", "#1b5e20", "#4CAF50", "#a5d6a7"])

    im = ax.imshow(cm_pct, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)

    # Add text annotations
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            count = cm[i, j]
            pct = cm_pct[i, j]
            text_color = "white" if pct > 50 else "#cccccc"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", color=text_color,
                    fontsize=12, fontweight="bold" if i == j else "normal")

    # Labels
    display_labels = [l.capitalize() for l in CLASS_LABELS]
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.set_yticklabels(display_labels, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(f"Confusion Matrix — Test Set (Accuracy: {overall_acc:.1%})",
                 fontsize=15, fontweight="bold", pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Classification Rate (%)", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    ✓ Saved: {save_path}")
    plt.close(fig)
    return True


# ============================================================
#  PLOT 4: SIMILARITY DISTRIBUTIONS
# ============================================================

def plot_similarity_distributions(fused_vectors, labels, label_names, profiles, save_path=None):
    """
    Compute cosine similarities between test embeddings and enrollment profiles.
    Plot genuine vs. impostor distributions with threshold lines.
    """
    print("\n  [4/6] Generating similarity distributions...")

    if profiles is None or fused_vectors is None:
        print("    ⚠ Skipped — profiles or test data not available")
        return False

    import torch.nn.functional as F

    genuine_sims = []
    impostor_sims = []

    for idx in range(fused_vectors.shape[0]):
        sample = fused_vectors[idx]
        true_label = label_names[idx]

        # Split fused vector back into face (512) and voice (192)
        face_emb = sample[:512]
        voice_emb = sample[512:]

        for person in AUTHORIZED_USERS:
            if person not in profiles:
                continue

            prof = profiles[person]
            ref_face = prof.get("face_mean", prof.get("face", None))
            ref_voice = prof.get("voice_mean", prof.get("voice", None))

            if ref_face is None or ref_voice is None:
                continue

            # Cosine similarity (average of face + voice)
            face_sim = F.cosine_similarity(
                face_emb.unsqueeze(0), ref_face.cpu().unsqueeze(0)
            ).item()
            voice_sim = F.cosine_similarity(
                voice_emb.unsqueeze(0), ref_voice.cpu().unsqueeze(0)
            ).item()
            avg_sim = (face_sim + voice_sim) / 2.0

            if true_label == person:
                genuine_sims.append(avg_sim)
            else:
                impostor_sims.append(avg_sim)

    if not genuine_sims or not impostor_sims:
        print("    ⚠ Skipped — not enough similarity data")
        return False

    print(f"    Genuine pairs:  {len(genuine_sims)}, mean={np.mean(genuine_sims):.3f}")
    print(f"    Impostor pairs: {len(impostor_sims)}, mean={np.mean(impostor_sims):.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    bins = np.linspace(-0.2, 1.0, 80)

    ax.hist(impostor_sims, bins=bins, alpha=0.6, color="#E53935", label="Impostor Pairs",
            edgecolor="#B71C1C", linewidth=0.5, density=True)
    ax.hist(genuine_sims, bins=bins, alpha=0.6, color="#43A047", label="Genuine Pairs",
            edgecolor="#1B5E20", linewidth=0.5, density=True)

    # Threshold lines
    ax.axvline(x=CONFIDENCE_HIGH, color="#FFD600", linestyle="--", linewidth=2,
               label=f"High Confidence ({CONFIDENCE_HIGH:.0%})")
    ax.axvline(x=CONFIDENCE_LOW, color="#FF6F00", linestyle="--", linewidth=2,
               label=f"Low Confidence ({CONFIDENCE_LOW:.0%})")

    # Annotate separation
    gen_mean = np.mean(genuine_sims)
    imp_mean = np.mean(impostor_sims)
    separation = gen_mean - imp_mean

    ax.annotate(f"Separation: {separation:.3f}",
                xy=((gen_mean + imp_mean) / 2, ax.get_ylim()[1] * 0.85),
                fontsize=11, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                          edgecolor="#FFD600", alpha=0.9))

    ax.set_title("Genuine vs. Impostor Similarity Distributions", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Average Cosine Similarity (Face + Voice)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", framealpha=0.8, facecolor="#1a1a2e", edgecolor="#555")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    ✓ Saved: {save_path}")
    plt.close(fig)
    return True


# ============================================================
#  PLOT 5: PER-CLASS PERFORMANCE BARS
# ============================================================

def plot_per_class_performance(fused_vectors, labels, model, save_path=None):
    """
    Grouped bar chart showing accuracy, precision, and recall per class.
    """
    print("\n  [5/6] Generating per-class performance bars...")

    if model is None or fused_vectors is None:
        print("    ⚠ Skipped — model or test data not available")
        return False

    from sklearn.metrics import precision_recall_fscore_support

    # Run inference
    with torch.no_grad():
        logits = model(fused_vectors)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

    y_true = labels.numpy()
    y_pred = preds.numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    x = np.arange(NUM_CLASSES)
    bar_width = 0.25

    bars_precision = ax.bar(x - bar_width, precision * 100, bar_width,
                            label="Precision", color="#2196F3", alpha=0.85,
                            edgecolor="white", linewidth=0.5)
    bars_recall = ax.bar(x, recall * 100, bar_width,
                         label="Recall", color="#FF9800", alpha=0.85,
                         edgecolor="white", linewidth=0.5)
    bars_f1 = ax.bar(x + bar_width, f1 * 100, bar_width,
                     label="F1-Score", color="#4CAF50", alpha=0.85,
                     edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bars in [bars_precision, bars_recall, bars_f1]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f"{height:.0f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    # Support counts below bars
    for i, count in enumerate(support):
        ax.text(i, -6, f"n={count}", ha="center", fontsize=9, alpha=0.7)

    display_labels = [l.capitalize() for l in CLASS_LABELS]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim([-10, 115])
    ax.set_title("Per-Class Classification Performance", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.8, facecolor="#1a1a2e", edgecolor="#555")
    ax.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    ✓ Saved: {save_path}")
    plt.close(fig)
    return True


# ============================================================
#  PLOT 6: SYSTEM ARCHITECTURE DIAGRAM
# ============================================================

def plot_system_architecture(save_path=None):
    """
    Draw a clean flowchart of the full authentication pipeline.
    """
    print("\n  [6/6] Generating system architecture diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    def draw_box(x, y, w, h, text, color="#16213e", border_color="#4CAF50",
                 text_color="#e0e0e0", fontsize=10, bold=False, radius=0.3):
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=color, edgecolor=border_color, linewidth=2
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight,
                multialignment="center")

    def draw_arrow(x1, y1, x2, y2, color="#e0e0e0"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    def draw_label(x, y, text, fontsize=9, color="#aaaaaa"):
        ax.text(x, y, text, fontsize=fontsize, color=color,
                ha="center", va="center", style="italic")

    # ---- Title ----
    ax.text(9, 10.5, "SECURE GATEWAY — Authentication Pipeline",
            fontsize=20, fontweight="bold", color="#e0e0e0",
            ha="center", va="center")

    # ---- User Input ----
    draw_box(0.3, 7.5, 2.5, 1.5, "👤 User\nApproaches\nSystem",
             color="#0d47a1", border_color="#42A5F5", fontsize=11, bold=True)

    draw_arrow(2.8, 8.25, 3.8, 8.25)

    # ---- Gate 1: Password ----
    draw_box(3.8, 7.5, 3.0, 1.5,
             "GATE 1\nVoice Password\n\"my voice is my password\"\nGoogle STT + Fuzzy Match",
             color="#1b3a1b", border_color="#66BB6A", fontsize=9, bold=True)

    draw_arrow(6.8, 8.25, 7.8, 8.25)
    draw_label(7.3, 8.7, "Pass ✓")

    # ---- Gate 2: Biometrics ----
    draw_box(7.8, 7.5, 3.0, 1.5,
             "GATE 2\nBiometric Capture\nFace: MTCNN → FaceNet (512d)\nVoice: ECAPA-TDNN (192d)",
             color="#1b2a3a", border_color="#42A5F5", fontsize=9, bold=True)

    draw_arrow(10.8, 8.25, 11.8, 8.25)

    # ---- Liveness Check ----
    draw_box(11.8, 7.5, 2.8, 1.5,
             "LIVENESS CHECK\nBlink Detection (EAR)\n+ Head Pose Variation\nMediaPipe Face Mesh",
             color="#3e1a1a", border_color="#EF5350", fontsize=9, bold=True)

    draw_arrow(14.6, 8.25, 15.5, 8.25)
    draw_label(15.05, 8.7, "Live ✓")

    # ---- Reject boxes ----
    # Password fail
    draw_box(4.3, 5.5, 2.0, 1.0, "🔴 DENIED\nPassword Failed",
             color="#4a1010", border_color="#E53935", fontsize=9)
    draw_arrow(5.3, 7.5, 5.3, 6.5)
    draw_label(4.7, 6.9, "Fail ✗")

    # Liveness fail
    draw_box(12.3, 5.5, 2.0, 1.0, "🔴 DENIED\nNot Live (Spoof)",
             color="#4a1010", border_color="#E53935", fontsize=9)
    draw_arrow(13.2, 7.5, 13.3, 6.5)
    draw_label(12.7, 6.9, "Fail ✗")

    # ---- Gate 3: Fusion Model ----
    draw_box(6.0, 3.0, 3.5, 1.8,
             "GATE 3 — FUSION MODEL\n704-dim → MLP → Softmax\n[David | Itzhak | Yossi | Unknown]",
             color="#1a1a3e", border_color="#7C4DFF", fontsize=10, bold=True)

    # Arrow down from liveness to fusion (goes down then left)
    ax.annotate("", xy=(7.75, 4.8), xytext=(15.5, 7.5),
                arrowprops=dict(arrowstyle="-|>", color="#e0e0e0", lw=2,
                                connectionstyle="arc3,rad=0.2"))

    # ---- Three-tier decision ----
    # High confidence
    draw_box(0.5, 0.5, 3.5, 1.5,
             "🟢 ACCESS GRANTED\nConfidence ≥ 85%\nImmediate Access",
             color="#1b3a1b", border_color="#66BB6A", fontsize=10, bold=True)

    # Gray area
    draw_box(5.0, 0.5, 4.5, 1.5,
             "🟡 GRAY AREA (50-85%)\nCosine Similarity Fallback\nFace + Voice vs. Profile\n→ Both > 0.4 = GRANT",
             color="#3a3a1b", border_color="#FFD600", fontsize=9, bold=True)

    # Low / Unknown
    draw_box(10.5, 0.5, 3.5, 1.5,
             "🔴 ACCESS DENIED\nConfidence < 50%\nor Unknown Class",
             color="#4a1010", border_color="#E53935", fontsize=10, bold=True)

    # Arrows from fusion to three tiers
    draw_arrow(6.5, 3.0, 2.25, 2.0)
    draw_arrow(7.75, 3.0, 7.25, 2.0)
    draw_arrow(9.0, 3.0, 12.25, 2.0)

    draw_label(3.8, 2.6, "≥ 85%")
    draw_label(7.5, 2.6, "50-85%")
    draw_label(11.0, 2.6, "< 50%")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    ✓ Saved: {save_path}")
    plt.close(fig)
    return True


# ============================================================
#  COMBINED DASHBOARD
# ============================================================

def create_dashboard(figure_paths, save_path=None):
    """
    Combine all 6 plots into a single large dashboard image.
    """
    print("\n  Creating combined dashboard...")

    from PIL import Image

    # Filter to only existing images
    existing = [p for p in figure_paths if p and os.path.exists(p)]

    if len(existing) < 2:
        print("    ⚠ Not enough plots for dashboard — need at least 2")
        return False

    # Load all images
    images = []
    for path in existing:
        img = Image.open(path)
        images.append(img)

    # Arrange in a 2×3 grid (or 3×2 depending on count)
    n = len(images)
    if n <= 4:
        cols, rows = 2, 2
    else:
        cols, rows = 3, 2

    # Find max dimensions per cell
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    # Resize all to uniform size
    target_w = min(max_w, 1800)
    target_h = min(max_h, 1200)

    resized = []
    for img in images:
        ratio = min(target_w / img.width, target_h / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        resized.append(img.resize((new_w, new_h), Image.LANCZOS))

    # Uniform cell size
    cell_w = max(img.width for img in resized)
    cell_h = max(img.height for img in resized)

    padding = 20
    title_h = 80

    dash_w = cols * cell_w + (cols + 1) * padding
    dash_h = rows * cell_h + (rows + 1) * padding + title_h

    dashboard = Image.new("RGB", (dash_w, dash_h), color=(26, 26, 46))

    # Paste images
    for idx, img in enumerate(resized):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding) + (cell_w - img.width) // 2
        y = title_h + padding + row * (cell_h + padding) + (cell_h - img.height) // 2
        dashboard.paste(img, (x, y))

    if save_path:
        dashboard.save(save_path, quality=95)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"    ✓ Saved: {save_path} ({size_mb:.1f} MB)")

    return True


# ============================================================
#  MAIN ENTRY POINT
# ============================================================

def main():
    """Run all visualizations and save to evaluation/figures/."""

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  SECURE GATEWAY — Visualization & Analysis Suite         " + "║")
    print("║" + "  Phase 5: Generating all plots for video presentation    " + "║")
    print("╚" + "═" * 58 + "╝")

    start_time = time.time()

    # Create output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\n  Output directory: {FIGURES_DIR}")

    # ---- Load all data ----
    print("\n" + "=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    face_data, voice_data = load_embeddings()
    splits_data = load_data_splits()
    history = load_training_history()
    model = load_fusion_model()
    profiles = load_user_profiles()

    # ---- Build test fused vectors ----
    print("\n  Building test fused vectors...")
    fused_vectors, labels, label_names = None, None, None

    if splits_data is not None and face_data is not None and voice_data is not None:
        fused_vectors, labels, label_names = build_fused_test_data(splits_data, face_data, voice_data)
        if fused_vectors is not None:
            print(f"    Built {fused_vectors.shape[0]} test pairs (from data splits)")
    elif face_data is not None and voice_data is not None:
        print("    ⚠ Using all embeddings (no splits available)")
        fused_vectors, labels, label_names = build_fused_from_raw(face_data, voice_data)
        if fused_vectors is not None:
            print(f"    Built {fused_vectors.shape[0]} pairs (from all embeddings)")

    if fused_vectors is not None:
        # Print class distribution
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        for cls_idx, count in zip(unique, counts):
            print(f"      {CLASS_LABELS[cls_idx]}: {count}")

    # ---- Generate all plots ----
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    figure_paths = []
    results = {}

    # Plot 1: t-SNE
    path1 = os.path.join(FIGURES_DIR, "1_tsne_clusters.png")
    if fused_vectors is not None:
        results["tsne"] = plot_tsne_clusters(fused_vectors, labels, label_names, profiles, save_path=path1)
        if results["tsne"]:
            figure_paths.append(path1)
    else:
        print("\n  [1/6] t-SNE — ⚠ Skipped (no embedding data)")

    # Plot 2: Training curves
    path2 = os.path.join(FIGURES_DIR, "2_training_curves.png")
    results["training"] = plot_training_curves(history, save_path=path2)
    if results["training"]:
        figure_paths.append(path2)

    # Plot 3: Confusion matrix
    path3 = os.path.join(FIGURES_DIR, "3_confusion_matrix.png")
    if fused_vectors is not None and model is not None:
        results["confusion"] = plot_confusion_matrix(fused_vectors, labels, model, save_path=path3)
        if results["confusion"]:
            figure_paths.append(path3)
    else:
        print("\n  [3/6] Confusion matrix — ⚠ Skipped (no model or test data)")

    # Plot 4: Similarity distributions
    path4 = os.path.join(FIGURES_DIR, "4_similarity_distributions.png")
    if fused_vectors is not None and profiles is not None:
        results["similarity"] = plot_similarity_distributions(
            fused_vectors, labels, label_names, profiles, save_path=path4)
        if results["similarity"]:
            figure_paths.append(path4)
    else:
        print("\n  [4/6] Similarity distributions — ⚠ Skipped (no profiles or test data)")

    # Plot 5: Per-class performance
    path5 = os.path.join(FIGURES_DIR, "5_per_class_performance.png")
    if fused_vectors is not None and model is not None:
        results["per_class"] = plot_per_class_performance(fused_vectors, labels, model, save_path=path5)
        if results["per_class"]:
            figure_paths.append(path5)
    else:
        print("\n  [5/6] Per-class performance — ⚠ Skipped (no model or test data)")

    # Plot 6: System architecture
    path6 = os.path.join(FIGURES_DIR, "6_system_architecture.png")
    results["architecture"] = plot_system_architecture(save_path=path6)
    if results["architecture"]:
        figure_paths.append(path6)

    # Dashboard
    dashboard_path = os.path.join(FIGURES_DIR, "dashboard.png")
    results["dashboard"] = create_dashboard(figure_paths, save_path=dashboard_path)

    # ---- Summary ----
    elapsed = time.time() - start_time
    generated = sum(1 for v in results.values() if v)
    total = len(results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Generated {generated}/{total} visualizations in {elapsed:.1f}s")
    print(f"  Output directory: {FIGURES_DIR}")
    print()

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"    {status} {name}")

    if figure_paths:
        print(f"\n  Files:")
        for path in figure_paths:
            size_kb = os.path.getsize(path) / 1024
            print(f"    → {os.path.basename(path)} ({size_kb:.0f} KB)")
        if os.path.exists(dashboard_path):
            size_kb = os.path.getsize(dashboard_path) / 1024
            print(f"    → dashboard.png ({size_kb:.0f} KB)")

    print(f"\n  Done!\n")


if __name__ == "__main__":
    main()
