"""
Fusion Model Training Pipeline
=================================
Trains the Late Fusion MLP that combines face + voice embeddings
to identify authorized users and reject impostors.

What this script does:
    1. Loads face and voice embeddings from .pt files
    2. Splits data per-person into train/val/test (BEFORE pairing — prevents leakage)
    3. Creates paired samples:
       - Genuine pairs: face_A + voice_A → label "A"  (same person)
       - Unknown pairs:  face_A + voice_B → label "unknown"  (cross-person)
    4. Handles class imbalance via weighted loss
    5. Trains the Late Fusion MLP with BatchNorm, Dropout, and early stopping
    6. Evaluates on held-out test set with per-class accuracy and confusion matrix
    7. Saves the best model + training history + metadata

Architecture:
    Input(704) → Linear(704→256) → BatchNorm → ReLU → Dropout(0.3)
               → Linear(256→128)  → BatchNorm → ReLU → Dropout(0.2)
               → Linear(128→4)    → (Softmax applied by CrossEntropyLoss)

    4 output classes: [david, itzhak, yossi, unknown]

Output files:
    models/fusion_model.pt     — Complete checkpoint (model weights + metadata)
    models/training_history.pt — Training curves (loss, accuracy per epoch)

Where to place this file:
    training/train_model.py

Usage:
    python training/train_model.py

Requirements:
    pip install torch numpy scikit-learn tqdm
    (Must run AFTER compute_embeddings.py has generated the .pt files)
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
#  Import project configuration
# ============================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    PATHS,
    CLASSES,
    EMBEDDINGS,
    TRAINING,
    DEVICE,
)


# ============================================================
#  STEP 0: Reproducibility — seed everything
# ============================================================

def set_all_seeds(seed):
    """
    Set random seeds for full reproducibility.

    Why seed everything?
        Without this, every training run produces different results because:
        - Data shuffling is random
        - Weight initialization is random
        - Dropout masks are random
        - Pair generation is random

        Setting seeds means: same data + same code = same model every time.
        This is critical for debugging and for scientific reproducibility.

    We seed 4 separate random number generators:
        1. Python's built-in random module (used for pair generation)
        2. NumPy's random (used for data splitting)
        3. PyTorch CPU random (used for weight init, dropout)
        4. PyTorch CUDA random (used for GPU operations)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)      # For multi-GPU setups
        # These two flags trade some speed for determinism on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"  Random seed set to {seed} across all generators.")


# ============================================================
#  STEP 1: Load and split embeddings
# ============================================================

def load_embeddings():
    """
    Load face and voice embedding files.

    Returns:
        face_data: dict {person: Tensor[N, 512]}
        voice_data: dict {person: Tensor[N, 192]}
    """
    print("\n  Loading embedding files...")

    if not os.path.exists(PATHS.FACE_EMBEDDINGS):
        print(f"  [ERROR] Not found: {PATHS.FACE_EMBEDDINGS}")
        print(f"  Run compute_embeddings.py first!")
        sys.exit(1)

    if not os.path.exists(PATHS.VOICE_EMBEDDINGS):
        print(f"  [ERROR] Not found: {PATHS.VOICE_EMBEDDINGS}")
        print(f"  Run compute_embeddings.py first!")
        sys.exit(1)

    face_data = torch.load(PATHS.FACE_EMBEDDINGS, map_location="cpu", weights_only=True)
    voice_data = torch.load(PATHS.VOICE_EMBEDDINGS, map_location="cpu", weights_only=True)

    for person in CLASSES.AUTHORIZED_USERS:
        if person not in face_data or person not in voice_data:
            print(f"  [ERROR] '{person}' missing from embeddings!")
            sys.exit(1)

    print(f"  ✓ Embeddings loaded.")
    for person in CLASSES.AUTHORIZED_USERS:
        print(f"    {person}: face={face_data[person].shape}, voice={voice_data[person].shape}")

    return face_data, voice_data


def split_embeddings(face_data, voice_data):
    """
    Split embeddings into train/val/test sets PER PERSON, PER MODALITY.

    WHY SPLIT BEFORE PAIRING:
        If we paired first and then split, the same face embedding could appear
        in a training pair AND a test pair (just with a different voice partner).
        The model would then be tested on data it's already seen → inflated accuracy.

        By splitting the raw embeddings first, we guarantee that NO face or voice
        embedding appears in more than one set.

    HOW:
        For each person, independently shuffle and split their face embeddings
        and their voice embeddings into 70/15/15 portions.

    SAVING:
        Both the shuffled indices AND the actual split tensors are saved to disk.
        This is critical because:
        - smart_finetune.py needs to know which embeddings are in the training set
          (to add corrected samples to the right set)
        - live_visualizer.py should only plot test embeddings for honest visualization
        - If you retrain with a different architecture, you need the exact same split
          to compare results fairly
        - The seed alone is NOT sufficient — if the embedding data changes (e.g.,
          you add more photos), the same seed produces different splits

    Args:
        face_data: dict {person: Tensor[N_face, 512]}
        voice_data: dict {person: Tensor[N_voice, 192]}

    Returns:
        splits: dict structured as:
            {
                "train": {"david": {"face": Tensor, "voice": Tensor}, ...},
                "val":   {"david": {"face": Tensor, "voice": Tensor}, ...},
                "test":  {"david": {"face": Tensor, "voice": Tensor}, ...},
            }
    """
    print(f"\n  Splitting embeddings (train={TRAINING.TRAIN_RATIO}/"
          f"val={TRAINING.VAL_RATIO}/test={TRAINING.TEST_RATIO})...")

    splits = {"train": {}, "val": {}, "test": {}}
    split_indices = {}  # Save indices for reproducibility

    for person in CLASSES.AUTHORIZED_USERS:
        face_tensor = face_data[person]
        voice_tensor = voice_data[person]

        # Shuffle indices independently for each modality
        # This ensures the split is random, not just taking the first N
        face_indices = torch.randperm(face_tensor.shape[0])
        voice_indices = torch.randperm(voice_tensor.shape[0])

        # Calculate split points for face
        n_face = face_tensor.shape[0]
        face_train_end = int(n_face * TRAINING.TRAIN_RATIO)
        face_val_end = face_train_end + int(n_face * TRAINING.VAL_RATIO)

        # Calculate split points for voice
        n_voice = voice_tensor.shape[0]
        voice_train_end = int(n_voice * TRAINING.TRAIN_RATIO)
        voice_val_end = voice_train_end + int(n_voice * TRAINING.VAL_RATIO)

        # Split face embeddings
        face_train = face_tensor[face_indices[:face_train_end]]
        face_val = face_tensor[face_indices[face_train_end:face_val_end]]
        face_test = face_tensor[face_indices[face_val_end:]]

        # Split voice embeddings
        voice_train = voice_tensor[voice_indices[:voice_train_end]]
        voice_val = voice_tensor[voice_indices[voice_train_end:voice_val_end]]
        voice_test = voice_tensor[voice_indices[voice_val_end:]]

        splits["train"][person] = {"face": face_train, "voice": voice_train}
        splits["val"][person] = {"face": face_val, "voice": voice_val}
        splits["test"][person] = {"face": face_test, "voice": voice_test}

        # Store the indices used for this split
        split_indices[person] = {
            "face_indices": face_indices,
            "voice_indices": voice_indices,
            "face_split_points": (face_train_end, face_val_end, n_face),
            "voice_split_points": (voice_train_end, voice_val_end, n_voice),
        }

        print(f"    {person}: train(f={face_train.shape[0]}, v={voice_train.shape[0]}) | "
              f"val(f={face_val.shape[0]}, v={voice_val.shape[0]}) | "
              f"test(f={face_test.shape[0]}, v={voice_test.shape[0]})")

    # Save the splits to disk
    save_splits(splits, split_indices)

    return splits


def save_splits(splits, split_indices):
    """
    Save the data splits and their indices to disk.

    Saves to: models/data_splits.pt (as defined in config.py)

    The file contains:
        - "splits": The actual train/val/test embedding tensors per person
        - "indices": The shuffled indices used to create the splits
        - "config": The split ratios and seed used (for reference)

    This allows other scripts to:
        1. Load the exact same splits without recomputing
        2. Know which embeddings belong to which set
        3. Add new samples to the correct set (smart_finetune.py)
    """
    os.makedirs(PATHS.MODELS_DIR, exist_ok=True)

    save_data = {
        "splits": splits,
        "indices": split_indices,
        "config": {
            "train_ratio": TRAINING.TRAIN_RATIO,
            "val_ratio": TRAINING.VAL_RATIO,
            "test_ratio": TRAINING.TEST_RATIO,
            "random_seed": TRAINING.RANDOM_SEED,
        },
    }

    torch.save(save_data, PATHS.DATA_SPLITS)
    file_size = os.path.getsize(PATHS.DATA_SPLITS) / 1024
    print(f"\n  ✓ Data splits saved to: {PATHS.DATA_SPLITS} ({file_size:.1f} KB)")


def load_splits(splits_path=None):
    """
    Load saved data splits from disk.

    This function is imported by other scripts:
        from training.train_model import load_splits
        split_data = load_splits()
        train_splits = split_data["splits"]["train"]

    Args:
        splits_path: Path to splits file (default: PATHS.DATA_SPLITS)

    Returns:
        dict with "splits", "indices", and "config" keys

    Raises:
        SystemExit if file not found
    """
    if splits_path is None:
        splits_path = PATHS.DATA_SPLITS

    if not os.path.exists(splits_path):
        print(f"  [ERROR] Data splits not found: {splits_path}")
        print(f"  Run train_model.py first!")
        sys.exit(1)

    split_data = torch.load(splits_path, map_location="cpu", weights_only=False)

    print(f"  ✓ Data splits loaded from: {splits_path}")
    print(f"    Split ratios: train={split_data['config']['train_ratio']}, "
          f"val={split_data['config']['val_ratio']}, "
          f"test={split_data['config']['test_ratio']}")

    return split_data


# ============================================================
#  STEP 2: Create paired samples (genuine + unknown)
# ============================================================

def create_genuine_pairs(person_data, person_name):
    """
    Create genuine (same-person) paired samples.

    PAIRING STRATEGY:
        We have more face embeddings than voice embeddings (~4:1 ratio).
        To use ALL face embeddings without wasting data:
          - For each face embedding, randomly pick a voice embedding from
            the same person to pair with.
          - This means each voice embedding gets reused ~4 times, but
            with DIFFERENT face partners each time → unique pairs.

        Why pair every face rather than every voice?
          - Face data has more variation (augmentation produced more samples)
          - Each face captures a slightly different angle/lighting
          - Reusing voice is fine because ECAPA-TDNN embeddings are already
            very robust — the model benefits more from face diversity.

    Args:
        person_data: dict {"face": Tensor[N_face, 512], "voice": Tensor[N_voice, 192]}
        person_name: str (e.g., "david")

    Returns:
        fused_pairs: Tensor[N_face, 704] — concatenated face+voice vectors
        labels: Tensor[N_face] — all set to person's class index
    """
    face_embs = person_data["face"]     # [N_face, 512]
    voice_embs = person_data["voice"]   # [N_voice, 192]
    n_face = face_embs.shape[0]
    n_voice = voice_embs.shape[0]

    # For each face embedding, pick a random voice embedding from the same person
    # torch.randint generates random indices into the voice array
    voice_indices = torch.randint(0, n_voice, (n_face,))

    # Concatenate: each row = [face_i | voice_random_j] where i,j are same person
    fused = torch.cat([face_embs, voice_embs[voice_indices]], dim=1)  # [N_face, 704]

    # Label = this person's class index
    label_idx = CLASSES.CLASS_TO_IDX[person_name]
    labels = torch.full((n_face,), label_idx, dtype=torch.long)

    return fused, labels


def create_unknown_pairs(split_data, target_count):
    """
    Create unknown/impostor (cross-person) paired samples.

    HOW:
        Take a face from person A and a voice from person B (where A ≠ B).
        The model has never seen this combination as valid → label as "unknown".

    WHY THIS WORKS FOR ANTI-SPOOFING:
        An attacker presenting a photo of David + recording of Yossi creates
        exactly this kind of cross-person pair. By training on thousands of
        these mismatches, the model learns that face-voice correlation matters
        and that mismatched pairs should activate the "unknown" neuron.

    BALANCE:
        We generate approximately `target_count` unknown pairs so the unknown
        class is balanced against the total genuine pairs. This prevents the
        model from being biased toward or against the unknown class.

    DIVERSITY:
        We cycle through ALL possible cross-person combinations:
        (david_face + itzhak_voice), (david_face + yossi_voice),
        (itzhak_face + david_voice), etc. — 6 combinations total.
        Pairs are distributed evenly across combinations.

    Args:
        split_data: dict {person: {"face": Tensor, "voice": Tensor}} for one split
        target_count: int — approximate number of unknown pairs to generate

    Returns:
        fused_pairs: Tensor[target_count, 704]
        labels: Tensor[target_count] — all set to unknown class index
    """
    users = CLASSES.AUTHORIZED_USERS
    unknown_idx = CLASSES.CLASS_TO_IDX[CLASSES.UNKNOWN_LABEL]

    # Generate all cross-person combinations
    cross_combinations = []
    for i, user_a in enumerate(users):
        for j, user_b in enumerate(users):
            if i != j:
                cross_combinations.append((user_a, user_b))
    # For 3 users: 6 combinations (A→B, A→C, B→A, B→C, C→A, C→B)

    # Distribute target_count evenly across combinations
    pairs_per_combo = target_count // len(cross_combinations)
    remainder = target_count % len(cross_combinations)

    all_fused = []
    all_labels = []

    for combo_idx, (face_user, voice_user) in enumerate(cross_combinations):
        face_embs = split_data[face_user]["face"]
        voice_embs = split_data[voice_user]["voice"]

        # How many pairs for this combination
        n_pairs = pairs_per_combo + (1 if combo_idx < remainder else 0)

        # Randomly sample face and voice indices
        face_indices = torch.randint(0, face_embs.shape[0], (n_pairs,))
        voice_indices = torch.randint(0, voice_embs.shape[0], (n_pairs,))

        # Concatenate mismatched face + voice
        fused = torch.cat([
            face_embs[face_indices],
            voice_embs[voice_indices]
        ], dim=1)  # [n_pairs, 704]

        labels = torch.full((n_pairs,), unknown_idx, dtype=torch.long)

        all_fused.append(fused)
        all_labels.append(labels)

    return torch.cat(all_fused, dim=0), torch.cat(all_labels, dim=0)


def build_paired_dataset(split_data, split_name):
    """
    Build a complete paired dataset (genuine + unknown) from one split.

    BALANCE STRATEGY:
        1. Create genuine pairs for each authorized user
        2. Count total genuine pairs
        3. Generate the same number of unknown pairs
        → Ensures ~50% genuine / ~50% unknown

        Within the genuine class, different users may have different counts
        (david has more data). This is handled by class_weights in the loss
        function, NOT by discarding data.

    Args:
        split_data: dict {person: {"face": ..., "voice": ...}}
        split_name: str ("train", "val", or "test") — for logging

    Returns:
        all_fused: Tensor[total_pairs, 704] — concatenated vectors
        all_labels: Tensor[total_pairs] — class indices
    """
    print(f"\n  Building {split_name} dataset...")

    fused_list = []
    label_list = []

    # --- Generate genuine pairs for each authorized user ---
    genuine_total = 0
    for person in CLASSES.AUTHORIZED_USERS:
        fused, labels = create_genuine_pairs(split_data[person], person)
        fused_list.append(fused)
        label_list.append(labels)

        genuine_total += fused.shape[0]
        print(f"    {person}: {fused.shape[0]} genuine pairs")

    # --- Generate unknown pairs (balanced against genuine total) ---
    unknown_fused, unknown_labels = create_unknown_pairs(split_data, genuine_total)
    fused_list.append(unknown_fused)
    label_list.append(unknown_labels)
    print(f"    unknown: {unknown_fused.shape[0]} cross-person pairs")

    # --- Combine everything ---
    all_fused = torch.cat(fused_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    total = all_fused.shape[0]
    print(f"    → Total {split_name} samples: {total} "
          f"({genuine_total} genuine + {unknown_fused.shape[0]} unknown)")

    return all_fused, all_labels


# ============================================================
#  STEP 3: PyTorch Dataset
# ============================================================

class FusionDataset(Dataset):
    """
    Simple PyTorch Dataset wrapping fused embedding pairs and labels.

    Why use a Dataset class?
        PyTorch's DataLoader requires a Dataset object. The DataLoader handles:
        - Automatic batching (groups of 64 samples)
        - Shuffling (randomize order each epoch — critical for training)
        - Parallel data loading (num_workers — speeds up on multi-core CPUs)

        Without this, we'd have to manually batch and shuffle, which is
        error-prone and slower.
    """

    def __init__(self, fused_embeddings, labels):
        """
        Args:
            fused_embeddings: Tensor[N, 704]
            labels: Tensor[N] — class indices (0=david, 1=itzhak, 2=yossi, 3=unknown)
        """
        self.embeddings = fused_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================
#  STEP 4: Model Architecture
# ============================================================

class FusionModel(nn.Module):
    """
    Late Fusion MLP for multi-modal biometric identification.

    Architecture:
        Input (704-dim fused vector)
          │
          ├─ Linear(704 → 256)
          ├─ BatchNorm1d(256)     ← normalizes activations to zero mean / unit variance
          ├─ ReLU()               ← non-linear activation
          ├─ Dropout(0.3)         ← randomly zeros 30% of neurons during training
          │
          ├─ Linear(256 → 128)
          ├─ BatchNorm1d(128)
          ├─ ReLU()
          ├─ Dropout(0.2)
          │
          └─ Linear(128 → 4)     ← one neuron per class (no activation — CrossEntropyLoss applies LogSoftmax)

    Why BatchNorm is ESSENTIAL here:
        Your enrollment output showed that face embeddings have std ~0.02 while
        voice embeddings have std ~16 — an 800x scale difference. Without BatchNorm,
        the 192 voice dimensions would dominate the 512 face dimensions simply because
        their raw numbers are bigger. BatchNorm normalizes each dimension to roughly
        zero mean and unit variance, so all 704 dimensions contribute equally.

    Why Dropout:
        With a small dataset (~2000-3000 pairs), the model can easily memorize
        the training data. Dropout forces the model to learn redundant representations
        (no single neuron is essential) → better generalization to unseen data.
        We use higher dropout (0.3) in the wider layer and lower (0.2) in the narrower one.

    Why NO Softmax in the model:
        PyTorch's CrossEntropyLoss internally applies LogSoftmax + NLLLoss.
        Adding Softmax here would apply it twice → wrong gradients → training fails.
        For inference, we manually apply Softmax to get probabilities.

    This class is imported by other scripts:
        from training.train_model import FusionModel
    """

    def __init__(self, input_dim=EMBEDDINGS.FUSED_EMBEDDING_DIM,
                 hidden1=TRAINING.HIDDEN_1,
                 hidden2=TRAINING.HIDDEN_2,
                 num_classes=CLASSES.NUM_CLASSES,
                 dropout1=TRAINING.DROPOUT_1,
                 dropout2=TRAINING.DROPOUT_2):
        super(FusionModel, self).__init__()

        self.network = nn.Sequential(
            # --- Hidden layer 1 ---
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),

            # --- Hidden layer 2 ---
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            # --- Output layer ---
            nn.Linear(hidden2, num_classes),
            # No Softmax — CrossEntropyLoss handles it
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor[batch_size, 704] — fused face+voice embedding

        Returns:
            Tensor[batch_size, 4] — raw logits (unnormalized scores per class)
        """
        return self.network(x)

    def predict_proba(self, x):
        """
        Get class probabilities (for inference, NOT training).

        Applies Softmax to convert logits → probabilities that sum to 1.
        Used by run_system.py during live operation.

        Args:
            x: Tensor[batch_size, 704]

        Returns:
            Tensor[batch_size, 4] — probabilities (0 to 1, sum to 1 per row)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# ============================================================
#  STEP 5: Compute class weights for imbalanced data
# ============================================================

def compute_class_weights(labels):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    WHY:
        Your data is imbalanced: david has 816 face embeddings vs itzhak's 464.
        Without weighting, the model optimizes more for david (more samples = more
        gradient updates) and may underperform on itzhak/yossi.

        Class weights tell the loss function: "when you get an itzhak sample wrong,
        penalize MORE than when you get a david sample wrong." This forces the
        model to pay equal attention to all classes.

    HOW:
        weight_c = total_samples / (num_classes × count_c)

        If david has 600 samples and itzhak has 300:
          weight_david  = 1800 / (4 × 600) = 0.75  (less weight — plenty of data)
          weight_itzhak = 1800 / (4 × 300) = 1.50  (more weight — less data)

    Args:
        labels: Tensor[N] — class indices for all training samples

    Returns:
        weights: Tensor[num_classes] — weight per class for CrossEntropyLoss
    """
    class_counts = torch.zeros(CLASSES.NUM_CLASSES)
    for c in range(CLASSES.NUM_CLASSES):
        class_counts[c] = (labels == c).sum().float()

    total = labels.shape[0]
    weights = total / (CLASSES.NUM_CLASSES * class_counts)

    # Handle edge case: if a class has 0 samples, set weight to 0
    # (shouldn't happen, but prevents division by zero)
    weights[class_counts == 0] = 0.0

    print(f"\n  Class weights (inverse frequency):")
    for c in range(CLASSES.NUM_CLASSES):
        class_name = CLASSES.IDX_TO_CLASS[c]
        print(f"    {class_name:<12} count={int(class_counts[c]):<6} weight={weights[c]:.4f}")

    return weights


# ============================================================
#  STEP 6: Training loop (one epoch)
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    One epoch = one complete pass through the entire training dataset.
    The data is processed in mini-batches of TRAINING.BATCH_SIZE.

    For each batch:
        1. Forward pass:  predictions = model(inputs)
        2. Compute loss:  loss = CrossEntropyLoss(predictions, true_labels)
        3. Backward pass: loss.backward() — computes gradients
        4. Update weights: optimizer.step() — adjusts weights to reduce loss
        5. Zero gradients: optimizer.zero_grad() — clears old gradients

    Args:
        model: FusionModel instance
        dataloader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss with class weights)
        optimizer: Adam optimizer
        device: torch.device (CPU or CUDA)

    Returns:
        avg_loss: float — mean loss across all batches
        accuracy: float — fraction of correct predictions (0 to 1)
    """
    model.train()  # Enable dropout and BatchNorm training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * inputs.size(0)  # Weighted by batch size
        _, predicted = torch.max(outputs, 1)           # Get predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
#  STEP 7: Validation / Test evaluation (one epoch, no gradients)
# ============================================================

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation or test data.

    Same as training, but:
        - model.eval() disables dropout (uses all neurons) and BatchNorm
          uses running statistics instead of batch statistics
        - torch.no_grad() disables gradient computation (saves memory, faster)
        - No optimizer.step() — we're just measuring, not learning

    Also collects per-class predictions for detailed accuracy reporting.

    Returns:
        avg_loss: float
        accuracy: float
        all_preds: Tensor — all predicted class indices
        all_labels: Tensor — all true class indices
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    avg_loss = running_loss / total
    accuracy = correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
#  STEP 8: Per-class accuracy and confusion matrix
# ============================================================

def print_per_class_accuracy(preds, labels, title=""):
    """
    Print accuracy for each class individually.

    Why this matters:
        Overall accuracy can be misleading. If the model gets 100% on david
        but 0% on itzhak, the overall accuracy could still be 50%+, which
        looks fine but means the system is completely broken for itzhak.

        Per-class accuracy reveals these hidden failures.

    Also prints precision and recall per class:
        - Precision: "Of all the times the model said 'david', how often was it right?"
        - Recall:    "Of all the actual david samples, how many did the model find?"
    """
    if title:
        print(f"\n  {title}")

    print(f"    {'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10} "
          f"{'Precision':>10} {'Recall':>10}")
    print(f"    {'-' * 60}")

    for c in range(CLASSES.NUM_CLASSES):
        class_name = CLASSES.IDX_TO_CLASS[c]

        # True positives, predicted positives, actual positives
        mask_true = (labels == c)
        mask_pred = (preds == c)

        actual_count = mask_true.sum().item()
        predicted_count = mask_pred.sum().item()
        correct_count = (mask_true & mask_pred).sum().item()

        # Accuracy = correct / actual (same as recall)
        accuracy = correct_count / actual_count if actual_count > 0 else 0.0

        # Precision = correct / predicted
        precision = correct_count / predicted_count if predicted_count > 0 else 0.0

        # Recall = correct / actual
        recall = accuracy

        print(f"    {class_name:<12} {correct_count:>8} {actual_count:>8} "
              f"{accuracy:>9.1%} {precision:>9.1%} {recall:>9.1%}")


def print_confusion_matrix(preds, labels):
    """
    Print a confusion matrix.

    How to read it:
        - Rows = actual class (ground truth)
        - Columns = predicted class (model's output)
        - Diagonal = correct predictions (higher is better)
        - Off-diagonal = errors (lower is better)

    Example:
                    pred_david  pred_itzhak  pred_yossi  pred_unknown
        act_david       95          2            1           2
        act_itzhak       1         90            3           6
        ...

    This tells you exactly WHERE the model makes mistakes.
    If david is often confused with itzhak, their embeddings might be too similar.
    """
    n = CLASSES.NUM_CLASSES
    matrix = torch.zeros(n, n, dtype=torch.long)

    for true_c, pred_c in zip(labels, preds):
        matrix[true_c.item()][pred_c.item()] += 1

    # Print header
    header = "            " + "".join(f"{CLASSES.IDX_TO_CLASS[c]:>10}" for c in range(n))
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {header}")
    print(f"  {'':>12}" + "-" * (10 * n))

    for true_c in range(n):
        row_name = CLASSES.IDX_TO_CLASS[true_c]
        row_vals = "".join(f"{matrix[true_c][pred_c].item():>10}" for pred_c in range(n))
        print(f"  {row_name:>12}{row_vals}")


# ============================================================
#  STEP 9: Full training pipeline
# ============================================================

def train_model():
    """
    Main training function — executes the complete pipeline.

    Flow:
        1. Set seeds for reproducibility
        2. Load embeddings
        3. Split into train/val/test per person
        4. Create paired datasets (genuine + unknown)
        5. Build DataLoaders
        6. Initialize model, optimizer, scheduler, loss function
        7. Training loop with validation monitoring
        8. Restore best model weights
        9. Final evaluation on test set
        10. Save model checkpoint + training history
    """
    print("=" * 60)
    print("  FUSION MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"\n  Device: {DEVICE.NAME}")
    print(f"  Architecture: {EMBEDDINGS.FUSED_EMBEDDING_DIM} → "
          f"{TRAINING.HIDDEN_1} → {TRAINING.HIDDEN_2} → {CLASSES.NUM_CLASSES}")
    print(f"  Classes: {CLASSES.ALL_CLASSES}")

    device = DEVICE.COMPUTE
    start_time = time.time()

    # ---- Step 1: Reproducibility ----
    set_all_seeds(TRAINING.RANDOM_SEED)

    # ---- Step 2: Load embeddings ----
    face_data, voice_data = load_embeddings()

    # ---- Step 3: Split data ----
    splits = split_embeddings(face_data, voice_data)

    # ---- Step 4: Create paired datasets ----
    train_fused, train_labels = build_paired_dataset(splits["train"], "train")
    val_fused, val_labels = build_paired_dataset(splits["val"], "val")
    test_fused, test_labels = build_paired_dataset(splits["test"], "test")

    # ---- Step 5: Build DataLoaders ----
    train_dataset = FusionDataset(train_fused, train_labels)
    val_dataset = FusionDataset(val_fused, val_labels)
    test_dataset = FusionDataset(test_fused, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING.BATCH_SIZE,
        shuffle=True,           # Shuffle training data each epoch — critical!
        drop_last=False,        # Keep the last incomplete batch
        num_workers=0,          # Single-threaded (safe on all platforms)
        pin_memory=torch.cuda.is_available(),  # Speeds up CPU→GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING.BATCH_SIZE,
        shuffle=False,          # No need to shuffle validation
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING.BATCH_SIZE,
        shuffle=False,
    )

    print(f"\n  DataLoaders ready:")
    print(f"    Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"    Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"    Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    # ---- Step 6: Initialize model + training components ----
    model = FusionModel().to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model initialized:")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Class-weighted loss function
    class_weights = compute_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Adam optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING.LEARNING_RATE,
        weight_decay=TRAINING.WEIGHT_DECAY,
    )

    # Learning rate scheduler — reduces LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",                              # Minimize validation loss
        patience=TRAINING.SCHEDULER_PATIENCE,    # Wait this many epochs before reducing
        factor=TRAINING.SCHEDULER_FACTOR,        # Multiply LR by this factor
        min_lr=1e-6,                             # Don't reduce below this
        verbose=False,
    )

    print(f"\n  Optimizer: Adam (lr={TRAINING.LEARNING_RATE}, "
          f"weight_decay={TRAINING.WEIGHT_DECAY})")
    print(f"  Scheduler: ReduceLROnPlateau (patience={TRAINING.SCHEDULER_PATIENCE}, "
          f"factor={TRAINING.SCHEDULER_FACTOR})")
    print(f"  Early stopping: patience={TRAINING.EARLY_STOPPING_PATIENCE} epochs")

    # ---- Step 7: Training loop ----
    print(f"\n  {'='*60}")
    print(f"  TRAINING — {TRAINING.EPOCHS} epochs max")
    print(f"  {'='*60}\n")

    # Track metrics for every epoch (used for visualization later)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    # Early stopping state
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, TRAINING.EPOCHS + 1):
        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # --- Validate ---
        val_loss, val_acc, val_preds, val_labels_out = evaluate(
            model, val_loader, criterion, device
        )

        # --- Update scheduler ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # --- Record history ---
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        # --- Check for improvement ---
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1

        # --- Print progress ---
        marker = " ★ best" if improved else ""
        lr_change = f" (lr→{new_lr:.2e})" if new_lr != current_lr else ""
        print(f"  Epoch {epoch:>3}/{TRAINING.EPOCHS} │ "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} │ "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1%}"
              f"{lr_change}{marker}")

        # --- Early stopping check ---
        if epochs_without_improvement >= TRAINING.EARLY_STOPPING_PATIENCE:
            print(f"\n  ⚠ Early stopping triggered! No improvement for "
                  f"{TRAINING.EARLY_STOPPING_PATIENCE} epochs.")
            print(f"  Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}, "
                  f"val_acc={best_val_acc:.1%}")
            break

    # ---- Step 8: Restore best model ----
    print(f"\n  Restoring best model from epoch {best_epoch}...")
    model.load_state_dict(best_model_state)

    # ---- Step 9: Final evaluation on TEST set ----
    print(f"\n  {'='*60}")
    print(f"  FINAL EVALUATION ON TEST SET")
    print(f"  {'='*60}")
    print(f"  (This data was NEVER seen during training or validation)")

    test_loss, test_acc, test_preds, test_labels_out = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%} ({(test_preds == test_labels_out).sum().item()}"
          f"/{len(test_labels_out)} correct)")

    print_per_class_accuracy(test_preds, test_labels_out, title="Per-Class Test Results:")
    print_confusion_matrix(test_preds, test_labels_out)

    # ---- Step 10: Save everything ----
    save_model(model, history, test_acc, best_epoch, best_val_acc,
               train_fused.shape[0], val_fused.shape[0], test_fused.shape[0])

    # ---- Final timing ----
    elapsed = time.time() - start_time
    print(f"\n  Total training time: {elapsed:.1f} seconds")

    print(f"\n  Files saved:")
    print(f"    models/fusion_model.pt      — trained model + metadata")
    print(f"    models/training_history.pt  — loss/accuracy per epoch")
    print(f"    models/data_splits.pt       — train/val/test split data")

    print(f"\n  Next step: python app/run_system.py")
    print(f"  (Run the live biometric security system)\n")

    return model, history


# ============================================================
#  STEP 10: Save model checkpoint
# ============================================================

def save_model(model, history, test_acc, best_epoch, best_val_acc,
               n_train, n_val, n_test):
    """
    Save the trained model with all metadata.

    WHY SAVE METADATA:
        When run_system.py loads the model, it needs to know:
        - The architecture parameters (to reconstruct the model)
        - The class mapping (to convert output index → person name)
        - The training config (for reference and debugging)

        Saving just the weights (state_dict) isn't enough — you'd need
        to hardcode the architecture elsewhere. By saving everything
        together, the model file is completely self-contained.

    WHY SAVE TRAINING HISTORY:
        The history (loss/accuracy per epoch) is used by:
        - live_visualizer.py to plot training curves
        - You, to diagnose issues (was there overfitting? did LR schedule help?)

    Files saved:
        models/fusion_model.pt      — model checkpoint + metadata
        models/training_history.pt  — epoch-by-epoch metrics
        models/data_splits.pt       — saved by split_embeddings() earlier in the pipeline
    """
    os.makedirs(PATHS.MODELS_DIR, exist_ok=True)

    # --- Save model checkpoint ---
    checkpoint = {
        # The model's learned weights (the actual "brain")
        "model_state_dict": model.state_dict(),

        # Architecture parameters (needed to reconstruct the model)
        "model_config": {
            "input_dim": EMBEDDINGS.FUSED_EMBEDDING_DIM,
            "hidden1": TRAINING.HIDDEN_1,
            "hidden2": TRAINING.HIDDEN_2,
            "num_classes": CLASSES.NUM_CLASSES,
            "dropout1": TRAINING.DROPOUT_1,
            "dropout2": TRAINING.DROPOUT_2,
        },

        # Class mapping (needed to interpret model output)
        "class_to_idx": CLASSES.CLASS_TO_IDX,
        "idx_to_class": CLASSES.IDX_TO_CLASS,
        "authorized_users": CLASSES.AUTHORIZED_USERS,
        "unknown_label": CLASSES.UNKNOWN_LABEL,

        # Training metadata
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "training_config": {
            "learning_rate": TRAINING.LEARNING_RATE,
            "weight_decay": TRAINING.WEIGHT_DECAY,
            "batch_size": TRAINING.BATCH_SIZE,
            "epochs_trained": best_epoch,
            "max_epochs": TRAINING.EPOCHS,
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "random_seed": TRAINING.RANDOM_SEED,
        },
    }

    torch.save(checkpoint, PATHS.FUSION_MODEL)
    model_size = os.path.getsize(PATHS.FUSION_MODEL) / 1024
    print(f"\n  Model saved to: {PATHS.FUSION_MODEL} ({model_size:.1f} KB)")

    # --- Save training history ---
    history_path = os.path.join(PATHS.MODELS_DIR, "training_history.pt")
    torch.save(history, history_path)
    print(f"  History saved to: {history_path}")

    # --- Print summary ---
    print(f"\n  Checkpoint contents:")
    print(f"    • model_state_dict — Trained weights ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"    • model_config     — Architecture (to rebuild the model)")
    print(f"    • class_to_idx     — {CLASSES.CLASS_TO_IDX}")
    print(f"    • idx_to_class     — {CLASSES.IDX_TO_CLASS}")
    print(f"    • best_epoch       — {best_epoch}")
    print(f"    • best_val_acc     — {best_val_acc:.1%}")
    print(f"    • test_acc         — {test_acc:.1%}")


# ============================================================
#  UTILITY: Load a saved model (used by other scripts)
# ============================================================

def load_trained_model(model_path=None, device=None):
    """
    Load a trained FusionModel from a checkpoint file.

    This function is imported by other scripts:
        from training.train_model import load_trained_model
        model, metadata = load_trained_model()

    It reconstructs the model architecture from the saved config,
    loads the weights, and returns both the model and metadata.

    Args:
        model_path: Path to the .pt checkpoint file (default: PATHS.FUSION_MODEL)
        device: torch.device (default: DEVICE.COMPUTE)

    Returns:
        model: FusionModel with loaded weights, in eval mode
        metadata: dict with class mappings, training info, etc.
    """
    if model_path is None:
        model_path = PATHS.FUSION_MODEL
    if device is None:
        device = DEVICE.COMPUTE

    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found: {model_path}")
        print(f"  Run train_model.py first!")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Reconstruct model from saved config
    config = checkpoint["model_config"]
    model = FusionModel(
        input_dim=config["input_dim"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        num_classes=config["num_classes"],
        dropout1=config["dropout1"],
        dropout2=config["dropout2"],
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Prepare metadata for the caller
    metadata = {
        "class_to_idx": checkpoint["class_to_idx"],
        "idx_to_class": checkpoint["idx_to_class"],
        "authorized_users": checkpoint["authorized_users"],
        "unknown_label": checkpoint["unknown_label"],
        "test_acc": checkpoint.get("test_acc", None),
        "best_val_acc": checkpoint.get("best_val_acc", None),
        "best_epoch": checkpoint.get("best_epoch", None),
        "training_config": checkpoint.get("training_config", {}),
    }

    print(f"  ✓ Model loaded from: {model_path}")
    print(f"    Test accuracy: {metadata['test_acc']:.1%}" if metadata["test_acc"] else "")
    print(f"    Classes: {list(metadata['class_to_idx'].keys())}")

    return model, metadata


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    train_model()
