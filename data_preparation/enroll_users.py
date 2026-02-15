"""
User Enrollment Pipeline
==========================
Takes the computed embeddings (face_embeddings.pt + voice_embeddings.pt)
and generates a single reference profile per user per modality.

What this script does:
    1. Loads all face embeddings and voice embeddings from the .pt files
    2. Validates that all expected users are present and dimensions are correct
    3. For each authorized user, computes:
       - Mean face embedding (512-dim)     → the user's "face fingerprint"
       - Mean voice embedding (192-dim)    → the user's "voice fingerprint"
       - Mean fused embedding (704-dim)    → concatenated face+voice (for 3D visualizer)
       - Standard deviation per modality   → measures how much the user varies
       - Sample count per modality         → confidence in the profile
    4. L2-normalizes all mean embeddings   → required for cosine similarity to work correctly
    5. Saves everything to models/user_profiles.pt

Why L2-normalize?
    Cosine similarity between two vectors A and B is:
        cos_sim = dot(A, B) / (||A|| * ||B||)
    If we pre-normalize the profile vectors to unit length (||profile|| = 1),
    then at live time we only need to normalize the incoming vector and compute
    a simple dot product — faster and numerically stable.

Why store standard deviation?
    The std tells us how much a person's embeddings naturally vary.
    In run_system.py, if a live embedding is many standard deviations away
    from the mean, that's an extra signal that something is off — even if
    cosine similarity alone looks borderline.

Output file structure (models/user_profiles.pt):
    {
        "david": {
            "face_mean":    Tensor[512],    # L2-normalized mean face embedding
            "voice_mean":   Tensor[192],    # L2-normalized mean voice embedding
            "fused_mean":   Tensor[704],    # L2-normalized concatenated mean
            "face_std":     Tensor[512],    # Per-dimension standard deviation
            "voice_std":    Tensor[192],    # Per-dimension standard deviation
            "face_count":   int,            # Number of face embeddings used
            "voice_count":  int,            # Number of voice embeddings used
        },
        "itzhak": { ... },
        "yossi":  { ... },
    }

Where to place this file:
    data_preparation/enroll_users.py

Usage:
    python data_preparation/enroll_users.py

Requirements:
    pip install torch numpy
    (Must run AFTER compute_embeddings.py has generated the .pt files)
"""

import os
import sys
import torch
import numpy as np

# ============================================================
#  Import project configuration
# ============================================================
# Add project root to Python path so we can import utils.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    PATHS,
    CLASSES,
    EMBEDDINGS,
    ENROLLMENT,
    DEVICE,
)


# ============================================================
#  STEP 1: Load and validate embeddings
# ============================================================

def load_embeddings():
    """
    Load the face and voice embedding files from disk.

    Expected format (produced by compute_embeddings.py):
        face_embeddings.pt  → {"david": Tensor[N, 512], "itzhak": ..., "yossi": ...}
        voice_embeddings.pt → {"david": Tensor[N, 192], "itzhak": ..., "yossi": ...}

    Returns:
        face_data (dict): {person_name: Tensor[N, 512]}
        voice_data (dict): {person_name: Tensor[N, 192]}

    Raises:
        SystemExit if files are missing or corrupted.
    """
    print("\n  Loading embedding files...")

    # --- Check that files exist ---
    if not os.path.exists(PATHS.FACE_EMBEDDINGS):
        print(f"\n  [ERROR] Face embeddings not found: {PATHS.FACE_EMBEDDINGS}")
        print(f"  Run compute_embeddings.py first!")
        sys.exit(1)

    if not os.path.exists(PATHS.VOICE_EMBEDDINGS):
        print(f"\n  [ERROR] Voice embeddings not found: {PATHS.VOICE_EMBEDDINGS}")
        print(f"  Run compute_embeddings.py first!")
        sys.exit(1)

    # --- Load the .pt files ---
    # map_location ensures it loads on CPU even if saved from GPU
    face_data = torch.load(PATHS.FACE_EMBEDDINGS, map_location="cpu", weights_only=True)
    voice_data = torch.load(PATHS.VOICE_EMBEDDINGS, map_location="cpu", weights_only=True)

    print(f"  ✓ Face embeddings loaded:  {PATHS.FACE_EMBEDDINGS}")
    print(f"  ✓ Voice embeddings loaded: {PATHS.VOICE_EMBEDDINGS}")

    return face_data, voice_data


def validate_embeddings(face_data, voice_data):
    """
    Verify that the loaded embeddings are valid and complete.

    Checks performed:
        1. All authorized users (from config) are present in both files
        2. Embedding dimensions match expected values (512 for face, 192 for voice)
        3. No empty tensors (every user has at least 1 embedding)
        4. No NaN or Inf values (corrupted data)

    Args:
        face_data: dict from face_embeddings.pt
        voice_data: dict from voice_embeddings.pt

    Returns:
        True if all checks pass

    Raises:
        SystemExit if any critical check fails
    """
    print("\n  Validating embeddings...")
    all_ok = True

    for person in CLASSES.AUTHORIZED_USERS:

        # --- Check person exists in both files ---
        if person not in face_data:
            print(f"  [ERROR] '{person}' not found in face embeddings!")
            print(f"  Available keys: {list(face_data.keys())}")
            all_ok = False
            continue

        if person not in voice_data:
            print(f"  [ERROR] '{person}' not found in voice embeddings!")
            print(f"  Available keys: {list(voice_data.keys())}")
            all_ok = False
            continue

        face_tensor = face_data[person]
        voice_tensor = voice_data[person]

        # --- Check dimensions ---
        if face_tensor.dim() != 2 or face_tensor.shape[1] != EMBEDDINGS.FACE_EMBEDDING_DIM:
            print(f"  [ERROR] '{person}' face embeddings have wrong shape: "
                  f"{face_tensor.shape}, expected [N, {EMBEDDINGS.FACE_EMBEDDING_DIM}]")
            all_ok = False

        if voice_tensor.dim() != 2 or voice_tensor.shape[1] != EMBEDDINGS.VOICE_EMBEDDING_DIM:
            print(f"  [ERROR] '{person}' voice embeddings have wrong shape: "
                  f"{voice_tensor.shape}, expected [N, {EMBEDDINGS.VOICE_EMBEDDING_DIM}]")
            all_ok = False

        # --- Check not empty ---
        if face_tensor.shape[0] == 0:
            print(f"  [ERROR] '{person}' has 0 face embeddings!")
            all_ok = False

        if voice_tensor.shape[0] == 0:
            print(f"  [ERROR] '{person}' has 0 voice embeddings!")
            all_ok = False

        # --- Check for NaN / Inf ---
        if torch.isnan(face_tensor).any():
            nan_count = torch.isnan(face_tensor).sum().item()
            print(f"  [ERROR] '{person}' face embeddings contain {nan_count} NaN values!")
            all_ok = False

        if torch.isinf(face_tensor).any():
            inf_count = torch.isinf(face_tensor).sum().item()
            print(f"  [ERROR] '{person}' face embeddings contain {inf_count} Inf values!")
            all_ok = False

        if torch.isnan(voice_tensor).any():
            nan_count = torch.isnan(voice_tensor).sum().item()
            print(f"  [ERROR] '{person}' voice embeddings contain {nan_count} NaN values!")
            all_ok = False

        if torch.isinf(voice_tensor).any():
            inf_count = torch.isinf(voice_tensor).sum().item()
            print(f"  [ERROR] '{person}' voice embeddings contain {inf_count} Inf values!")
            all_ok = False

        # --- Print summary for this person ---
        if all_ok:
            print(f"  ✓ {person}: face={face_tensor.shape}, voice={voice_tensor.shape}")

    if not all_ok:
        print(f"\n  [FATAL] Validation failed. Fix the issues above and re-run.")
        sys.exit(1)

    print(f"  ✓ All {len(CLASSES.AUTHORIZED_USERS)} users validated successfully.")
    return True


# ============================================================
#  STEP 2: Compute profiles
# ============================================================

def l2_normalize(tensor):
    """
    L2-normalize a vector so its length (magnitude) equals 1.

    After normalization:  ||tensor|| = 1.0
    This means cosine similarity simplifies to a dot product:
        cos_sim(a, b) = dot(a, b)   when both are unit vectors

    Args:
        tensor: 1D tensor of any dimension

    Returns:
        L2-normalized tensor (same shape, unit length)
    """
    norm = torch.norm(tensor, p=2)

    # Guard against zero-norm (would cause division by zero)
    # This should never happen with real embeddings, but just in case
    if norm < 1e-12:
        print(f"  [WARNING] Near-zero norm detected ({norm:.2e}), skipping normalization")
        return tensor

    return tensor / norm


def compute_single_profile(person_name, face_embeddings, voice_embeddings):
    """
    Compute the enrollment profile for a single person.

    This is the core computation:
        1. Mean embedding = centroid of all that person's embeddings
           → Represents their "average" face/voice in embedding space
           → Averaging across hundreds of samples (including augmented variations)
             produces a robust point that's resistant to noise

        2. Standard deviation = how spread out their embeddings are
           → A person with consistent photos will have low std
           → A person with varied conditions will have higher std
           → Useful later for anomaly detection

        3. L2 normalization = scale the mean to unit length
           → Required for cosine similarity to work as a simple dot product

        4. Fused embedding = concatenation of face + voice means
           → Used by the 3D visualizer (Phase 5) to plot user centroids

    Args:
        person_name: Name string (e.g., "david")
        face_embeddings: Tensor[N_face, 512]
        voice_embeddings: Tensor[N_voice, 192]

    Returns:
        dict with all profile components
    """
    # --- Compute means ---
    face_mean = face_embeddings.mean(dim=0)     # Tensor[512]
    voice_mean = voice_embeddings.mean(dim=0)   # Tensor[192]

    # --- Compute standard deviations ---
    # dim=0 computes std across all samples for each dimension
    # If only 1 sample, std would be 0 — that's fine, just means no variance info
    if face_embeddings.shape[0] > 1:
        face_std = face_embeddings.std(dim=0)   # Tensor[512]
    else:
        face_std = torch.zeros(EMBEDDINGS.FACE_EMBEDDING_DIM)

    if voice_embeddings.shape[0] > 1:
        voice_std = voice_embeddings.std(dim=0)  # Tensor[192]
    else:
        voice_std = torch.zeros(EMBEDDINGS.VOICE_EMBEDDING_DIM)

    # --- L2 normalize the means ---
    # This is critical for cosine similarity later
    face_mean_normalized = l2_normalize(face_mean)
    voice_mean_normalized = l2_normalize(voice_mean)

    # --- Create the fused (concatenated) mean ---
    # Used by the 3D visualizer to show each person's centroid
    fused_mean = torch.cat([face_mean_normalized, voice_mean_normalized])  # Tensor[704]
    fused_mean_normalized = l2_normalize(fused_mean)

    # --- Build the profile dict ---
    profile = {
        "face_mean":    face_mean_normalized,       # Tensor[512], unit length
        "voice_mean":   voice_mean_normalized,      # Tensor[192], unit length
        "fused_mean":   fused_mean_normalized,      # Tensor[704], unit length
        "face_std":     face_std,                   # Tensor[512], raw std values
        "voice_std":    voice_std,                  # Tensor[192], raw std values
        "face_count":   face_embeddings.shape[0],   # int, number of face samples
        "voice_count":  voice_embeddings.shape[0],  # int, number of voice samples
    }

    return profile


def compute_all_profiles(face_data, voice_data):
    """
    Compute enrollment profiles for all authorized users.

    Iterates over CLASSES.AUTHORIZED_USERS and computes a profile for each.
    Also runs sanity checks on the resulting profiles:
        - Are the profiles actually different from each other?
        - Are the cosine similarities between different users low?

    Args:
        face_data: dict from face_embeddings.pt
        voice_data: dict from voice_embeddings.pt

    Returns:
        profiles: dict {person_name: profile_dict}
    """
    print("\n  Computing enrollment profiles...")
    print(f"  Aggregation method: {ENROLLMENT.AGGREGATION_METHOD}")

    profiles = {}

    for person in CLASSES.AUTHORIZED_USERS:
        profile = compute_single_profile(
            person,
            face_data[person],
            voice_data[person],
        )
        profiles[person] = profile

        # Print profile summary
        face_norm = torch.norm(profile["face_mean"]).item()
        voice_norm = torch.norm(profile["voice_mean"]).item()

        print(f"\n  Profile for '{person}':")
        print(f"    Face:  mean computed from {profile['face_count']} embeddings "
              f"(norm={face_norm:.4f}, avg_std={profile['face_std'].mean():.4f})")
        print(f"    Voice: mean computed from {profile['voice_count']} embeddings "
              f"(norm={voice_norm:.4f}, avg_std={profile['voice_std'].mean():.4f})")

    return profiles


# ============================================================
#  STEP 3: Sanity checks — verify profiles are distinct
# ============================================================

def verify_profile_separation(profiles):
    """
    Check that the profiles for different users are actually distinct.

    If two users have very similar profiles (cosine similarity > 0.9),
    it means the system won't be able to reliably tell them apart.
    This could happen if:
        - Two people look very similar
        - The face images got mixed up between folders
        - The augmentation created too much noise

    We check both face-to-face and voice-to-voice similarity between
    all pairs of users. We also compute the fused similarity.

    This is a WARNING check, not a hard failure — the system might still
    work, but accuracy will likely suffer.
    """
    print("\n  Verifying profile separation (cross-user similarity)...")
    print(f"  Lower similarity = better separation = easier for the model\n")

    users = CLASSES.AUTHORIZED_USERS
    all_good = True

    # Header
    print(f"    {'Pair':<20} {'Face Sim':>10} {'Voice Sim':>10} {'Fused Sim':>10}    Status")
    print(f"    {'-' * 70}")

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user_a = users[i]
            user_b = users[j]

            # Cosine similarity between L2-normalized vectors = dot product
            face_sim = torch.dot(
                profiles[user_a]["face_mean"],
                profiles[user_b]["face_mean"]
            ).item()

            voice_sim = torch.dot(
                profiles[user_a]["voice_mean"],
                profiles[user_b]["voice_mean"]
            ).item()

            fused_sim = torch.dot(
                profiles[user_a]["fused_mean"],
                profiles[user_b]["fused_mean"]
            ).item()

            # Determine status
            if face_sim > 0.9 or voice_sim > 0.9:
                status = "⚠ WARNING: very similar!"
                all_good = False
            elif face_sim > 0.7 or voice_sim > 0.7:
                status = "⚠ moderate similarity"
            else:
                status = "✓ well separated"

            pair_label = f"{user_a} vs {user_b}"
            print(f"    {pair_label:<20} {face_sim:>10.4f} {voice_sim:>10.4f} {fused_sim:>10.4f}    {status}")

    print()
    if all_good:
        print(f"  ✓ All user profiles are well separated.")
    else:
        print(f"  ⚠ Some profiles are very similar. This may cause confusion.")
        print(f"    Consider: more diverse photos, checking for mislabeled data,")
        print(f"    or adjusting augmentation parameters.")

    return all_good


# ============================================================
#  STEP 4: Save profiles
# ============================================================

def save_profiles(profiles):
    """
    Save the enrollment profiles to disk.

    Saved to: models/user_profiles.pt (as defined in config.py)

    The file is a dict of dicts:
        {
            "david":  {"face_mean": ..., "voice_mean": ..., ...},
            "itzhak": {"face_mean": ..., "voice_mean": ..., ...},
            "yossi":  {"face_mean": ..., "voice_mean": ..., ...},
        }

    Later scripts load it with:
        profiles = torch.load("models/user_profiles.pt")
        david_face = profiles["david"]["face_mean"]
    """
    # Ensure output directory exists
    os.makedirs(PATHS.MODELS_DIR, exist_ok=True)

    torch.save(profiles, PATHS.USER_PROFILES)

    print(f"\n  Profiles saved to: {PATHS.USER_PROFILES}")

    # Print file size
    file_size = os.path.getsize(PATHS.USER_PROFILES)
    if file_size < 1024:
        print(f"  File size: {file_size} bytes")
    else:
        print(f"  File size: {file_size / 1024:.1f} KB")


# ============================================================
#  STEP 5: Run the full enrollment pipeline
# ============================================================

def run_enrollment():
    """
    Main function — executes the complete enrollment pipeline.

    Flow:
        1. Load embedding files
        2. Validate all users and dimensions
        3. Compute mean profiles per user
        4. Verify profiles are distinct from each other
        5. Save to models/user_profiles.pt
    """
    print("=" * 60)
    print("  USER ENROLLMENT PIPELINE")
    print("=" * 60)
    print(f"\n  Device: {DEVICE.NAME}")
    print(f"  Authorized users: {CLASSES.AUTHORIZED_USERS}")

    # --- Step 1: Load ---
    face_data, voice_data = load_embeddings()

    # --- Step 2: Validate ---
    validate_embeddings(face_data, voice_data)

    # --- Step 3: Compute profiles ---
    profiles = compute_all_profiles(face_data, voice_data)

    # --- Step 4: Sanity check — are profiles distinct? ---
    verify_profile_separation(profiles)

    # --- Step 5: Save ---
    save_profiles(profiles)

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("  ENROLLMENT COMPLETE — SUMMARY")
    print("=" * 60)

    print(f"\n  {'User':<15} {'Face Samples':<15} {'Voice Samples':<15} {'Fused Dim'}")
    print(f"  {'-' * 55}")
    for person in CLASSES.AUTHORIZED_USERS:
        p = profiles[person]
        fused_dim = p["fused_mean"].shape[0]
        print(f"  {person:<15} {p['face_count']:<15} {p['voice_count']:<15} {fused_dim}")

    print(f"\n  Output: {PATHS.USER_PROFILES}")
    print(f"\n  What's stored per user:")
    print(f"    • face_mean   — L2-normalized mean face embedding ({EMBEDDINGS.FACE_EMBEDDING_DIM}-dim)")
    print(f"    • voice_mean  — L2-normalized mean voice embedding ({EMBEDDINGS.VOICE_EMBEDDING_DIM}-dim)")
    print(f"    • fused_mean  — L2-normalized concatenated mean ({EMBEDDINGS.FUSED_EMBEDDING_DIM}-dim)")
    print(f"    • face_std    — Per-dimension standard deviation (face)")
    print(f"    • voice_std   — Per-dimension standard deviation (voice)")
    print(f"    • face_count  — Number of face samples used")
    print(f"    • voice_count — Number of voice samples used")

    print(f"\n  These profiles are used by:")
    print(f"    • run_system.py      — Cosine similarity fallback (gray area)")
    print(f"    • live_visualizer.py — 3D plot centroids")
    print(f"    • smart_finetune.py  — Profile update after admin correction")

    print(f"\n  Next step: python training/train_model.py")
    print(f"  (Train the Fusion Model using the embeddings)\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    run_enrollment()
