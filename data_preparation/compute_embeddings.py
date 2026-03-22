"""
Embedding Computation Pipeline
================================
Takes augmented face images and voice clips, passes them through pretrained
neural networks, and saves the resulting embedding vectors to disk.

What this script does:
    1. Loads all face images from data/face/augmented/<person>/
       → Passes through InceptionResnetV1 (FaceNet, pretrained on VGGFace2)
       → Outputs 512-dimensional face embedding vectors
       → Saves to data/embeddings/face_embeddings.pt

    2. Loads all voice clips from data/voice/augmented/<person>/
       → Passes through ECAPA-TDNN (pretrained on VoxCeleb via SpeechBrain)
       → Outputs 192-dimensional voice embedding vectors
       → Saves to data/embeddings/voice_embeddings.pt

Why two separate models?
    Each modality (face, voice) has fundamentally different data:
    - Face: 2D pixel grid (160×160×3 = 76,800 values per image)
    - Voice: 1D waveform (16000 × seconds = variable length)
    
    Each model was trained on millions of examples of its own modality
    and learned to compress the identity-relevant information into a
    compact vector (embedding). We leverage this via transfer learning —
    using their knowledge without retraining.

Output file structure:
    face_embeddings.pt:
    {
        "david": Tensor[480, 512],     # 480 images × 512-dim embedding
        "itzhak": Tensor[480, 512],
        "yossi": Tensor[480, 512],
    }
    
    voice_embeddings.pt:
    {
        "david": Tensor[120, 192],     # 120 clips × 192-dim embedding
        "itzhak": Tensor[120, 192],
        "yossi": Tensor[120, 192],
    }

Usage:
    python data_preparation/compute_embeddings.py

Requirements:
    pip install torch torchvision torchaudio facenet-pytorch speechbrain tqdm numpy
"""

import os
os.environ["SB_LOCAL_STRATEGY"] = "copy"

import sys
import torch
import torchaudio
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
#  CONFIGURATION
# ============================================================

# Input paths (augmented data — includes originals + augmented versions)
FACE_DATA_DIR = "data/face/augmented"
VOICE_DATA_DIR = "data/voice/augmented"

# Output paths
EMBEDDINGS_DIR = "data/embeddings"
FACE_EMBEDDINGS_FILE = "face_embeddings.pt"
VOICE_EMBEDDINGS_FILE = "voice_embeddings.pt"

# Processing settings
BATCH_SIZE = 32             # Images/clips per batch (adjust based on your GPU/RAM)
                             # 32 works well for most setups with 8GB+ RAM
                             # Lower to 16 if you run out of memory

# Device selection
# GPU (CUDA) is much faster but CPU works fine — just slower
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Voice settings
VOICE_SAMPLE_RATE = 16000    # Expected sample rate (must match preprocessing)


# ============================================================
#  PART 1: FACE EMBEDDINGS
# ============================================================

def load_face_model():
    """
    Load the pretrained InceptionResnetV1 (FaceNet) model.
    
    This model was trained on VGGFace2 — a dataset of 3.3 million face images
    of 9,131 people. It learned to map any face to a 512-dimensional vector
    where faces of the same person cluster together and different people
    are pushed apart.
    
    We use it in eval() mode with torch.no_grad() because we're NOT training —
    just extracting features. The model weights are frozen.
    
    Returns:
        model: InceptionResnetV1 ready for inference
    """
    from facenet_pytorch import InceptionResnetV1
    
    print("  Loading InceptionResnetV1 (FaceNet, pretrained on VGGFace2)...")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    print(f"  Model loaded on {DEVICE}. Output: 512-dim embedding per face.")
    
    return model


def preprocess_face_image(image_path):
    """
    Load and preprocess a single face image for FaceNet.
    
    FaceNet expects:
      - Tensor of shape (3, 160, 160)
      - Pixel values standardized: (pixel - 127.5) / 128.0
        This maps [0, 255] → [-1, 1] approximately
    
    The standardization is called "fixed_image_standardization" in facenet-pytorch.
    It's the exact normalization used during VGGFace2 training, so we must use
    the same one for consistent results.
    
    Args:
        image_path: Path to a 160×160 PNG/JPG face image
    
    Returns:
        Tensor of shape (3, 160, 160), standardized
        OR None if loading fails
    """
    from facenet_pytorch import fixed_image_standardization
    
    try:
        img = Image.open(image_path).convert("RGB")
        
        # Convert to numpy float32, then to tensor
        img_np = np.array(img).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (H,W,3) → (3,H,W)
        
        # Apply FaceNet's standardization: (x - 127.5) / 128.0
        img_tensor = fixed_image_standardization(img_tensor)
        
        return img_tensor
    except Exception as e:
        print(f"    [ERROR] Could not load {image_path}: {e}")
        return None


def compute_face_embeddings():
    """
    Process all face images and compute embeddings.
    
    Flow:
      1. Load the FaceNet model (once)
      2. For each person:
         a. Load all augmented face images
         b. Preprocess each image (standardize)
         c. Pass through the model in batches
         d. Collect all 512-dim embedding vectors
      3. Save everything to face_embeddings.pt
    
    Returns:
        dict: {person_name: Tensor[num_images, 512]}
    """
    print("\n" + "=" * 60)
    print("  COMPUTING FACE EMBEDDINGS")
    print("=" * 60)
    
    if not os.path.exists(FACE_DATA_DIR):
        print(f"\n  [ERROR] Face data directory not found: {FACE_DATA_DIR}")
        print(f"  Run augment_faces.py first!")
        return None
    
    model = load_face_model()
    
    # Find all person folders
    people = [
        d for d in sorted(os.listdir(FACE_DATA_DIR))
        if os.path.isdir(os.path.join(FACE_DATA_DIR, d))
        and not d.startswith(".")
    ]
    
    if not people:
        print(f"  [ERROR] No person folders found in {FACE_DATA_DIR}")
        return None
    
    print(f"\n  Found {len(people)} people: {', '.join(people)}")
    
    dataset = {}
    
    for person in people:
        person_dir = os.path.join(FACE_DATA_DIR, person)
        
        # Get all image files
        valid_ext = {".png", ".jpg", ".jpeg"}
        image_files = [
            f for f in sorted(os.listdir(person_dir))
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        
        if not image_files:
            print(f"\n  [WARNING] No images found for {person}")
            continue
        
        print(f"\n  Processing {len(image_files)} face images for '{person}'...")
        
        all_embeddings = []
        batch = []
        failed = 0
        
        for filename in tqdm(image_files, desc=f"  {person}", leave=True):
            image_path = os.path.join(person_dir, filename)
            tensor = preprocess_face_image(image_path)
            
            if tensor is None:
                failed += 1
                continue
            
            batch.append(tensor)
            
            # When batch is full, compute embeddings
            if len(batch) >= BATCH_SIZE:
                batch_tensor = torch.stack(batch).to(DEVICE)
                with torch.no_grad():
                    embeddings = model(batch_tensor)
                all_embeddings.append(embeddings.cpu())
                batch = []
        
        # Process remaining images in the last partial batch
        if batch:
            batch_tensor = torch.stack(batch).to(DEVICE)
            with torch.no_grad():
                embeddings = model(batch_tensor)
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches into one tensor
        if all_embeddings:
            dataset[person] = torch.cat(all_embeddings, dim=0)
            print(f"  → {person}: {dataset[person].shape[0]} embeddings "
                  f"(shape: {dataset[person].shape}), {failed} failed")
        else:
            print(f"  → [WARNING] No embeddings generated for {person}")
    
    return dataset


# ============================================================
#  PART 2: VOICE EMBEDDINGS
# ============================================================

def load_voice_model():
    """
    Load the pretrained ECAPA-TDNN model via SpeechBrain.
    
    ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
    in TDNN) was trained on VoxCeleb1+2 — a dataset of 7,000+ speakers
    extracted from YouTube interview videos.
    
    It produces a 192-dimensional embedding for any audio clip.
    Like FaceNet for faces, voices of the same speaker cluster together
    in this 192-dim space.
    
    SpeechBrain handles the full pipeline internally:
      Audio waveform → Mel features → ECAPA-TDNN → 192-dim vector
    
    Returns:
        model: SpeechBrain's EncoderClassifier ready for inference
    """
    from speechbrain.inference import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    print("  Loading ECAPA-TDNN (pretrained on VoxCeleb via SpeechBrain)...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/voice/ecapa_tdnn",       # Cache model files here
        run_opts={"device": str(DEVICE)},
        local_strategy=LocalStrategy.COPY,       # Use COPY instead of SYMLINK (Windows fix)
    )
    print(f"  Model loaded on {DEVICE}. Output: 192-dim embedding per clip.")
    
    return model


def compute_voice_embeddings_for_folder(model, folder_path):
    """
    Compute voice embeddings for all audio files in a single folder.
    
    SpeechBrain's encode_batch expects:
      - Tensor of shape (batch, num_samples) — raw 16kHz waveform
      - Lengths tensor of shape (batch,) — relative lengths (1.0 = full length)
    
    It handles all internal preprocessing (Mel spectrogram, etc.) automatically.
    
    Args:
        model: SpeechBrain EncoderClassifier
        folder_path: Path to folder containing .wav files
    
    Returns:
        list of embedding tensors, count of failures
    """
    valid_ext = {".wav", ".flac", ".mp3", ".ogg"}
    audio_files = [
        f for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    
    if not audio_files:
        return [], 0
    
    embeddings_list = []
    failed = 0
    
    for filename in tqdm(audio_files, desc=f"    {os.path.basename(folder_path)}", leave=False):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if needed
            if sr != VOICE_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, VOICE_SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # SpeechBrain expects (batch, samples) — remove channel dim
            waveform = waveform.squeeze(0)  # (1, samples) → (samples,)
            
            # Compute embedding
            # encode_batch expects (batch, samples) so we add batch dim
            with torch.no_grad():
                embedding = model.encode_batch(waveform.unsqueeze(0).to(DEVICE))
            
            # embedding shape: (1, 1, 192) → squeeze to (192,)
            embedding = embedding.squeeze().cpu()
            embeddings_list.append(embedding)
            
        except Exception as e:
            print(f"    [ERROR] Failed to process {filename}: {e}")
            failed += 1
    
    return embeddings_list, failed


def compute_voice_embeddings():
    """
    Process all voice clips and compute embeddings.
    
    Handles both category-based (password/short/long) and flat folder structures.
    All categories for a person are combined into one embedding tensor.
    
    Returns:
        dict: {person_name: Tensor[num_clips, 192]}
    """
    print("\n" + "=" * 60)
    print("  COMPUTING VOICE EMBEDDINGS")
    print("=" * 60)
    
    if not os.path.exists(VOICE_DATA_DIR):
        print(f"\n  [ERROR] Voice data directory not found: {VOICE_DATA_DIR}")
        print(f"  Run augment_voices.py first!")
        return None
    
    model = load_voice_model()
    
    # Find all person folders
    people = [
        d for d in sorted(os.listdir(VOICE_DATA_DIR))
        if os.path.isdir(os.path.join(VOICE_DATA_DIR, d))
        and not d.startswith(".")
    ]
    
    if not people:
        print(f"  [ERROR] No person folders found in {VOICE_DATA_DIR}")
        return None
    
    print(f"\n  Found {len(people)} people: {', '.join(people)}")
    
    dataset = {}
    
    for person in people:
        print(f"\n  Processing voice clips for '{person}'...")
        person_dir = os.path.join(VOICE_DATA_DIR, person)
        
        all_embeddings = []
        total_failed = 0
        
        # Check for category subfolders
        categories = ["password", "short", "long"]
        has_categories = any(
            os.path.isdir(os.path.join(person_dir, cat))
            for cat in categories
        )
        
        if has_categories:
            # Process each category subfolder
            for category in categories:
                cat_dir = os.path.join(person_dir, category)
                if not os.path.exists(cat_dir):
                    continue
                
                print(f"    Category: {category}")
                emb_list, failed = compute_voice_embeddings_for_folder(model, cat_dir)
                all_embeddings.extend(emb_list)
                total_failed += failed
        else:
            # Flat structure — all files directly in person folder
            emb_list, failed = compute_voice_embeddings_for_folder(model, person_dir)
            all_embeddings.extend(emb_list)
            total_failed += failed
        
        # Stack all embeddings into one tensor
        if all_embeddings:
            dataset[person] = torch.stack(all_embeddings)
            print(f"  → {person}: {dataset[person].shape[0]} embeddings "
                  f"(shape: {dataset[person].shape}), {total_failed} failed")
        else:
            print(f"  → [WARNING] No embeddings generated for {person}")
    
    return dataset


# ============================================================
#  PART 3: SAVE EMBEDDINGS
# ============================================================

def save_embeddings(dataset, filename, modality_name):
    """
    Save embedding dataset to a .pt file.
    
    We use torch.save which uses Python's pickle under the hood.
    The .pt file contains a dict mapping person names to tensors.
    
    Args:
        dataset: dict {person_name: Tensor[N, embedding_dim]}
        filename: Output filename (e.g., "face_embeddings.pt")
        modality_name: "Face" or "Voice" (for display)
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    output_path = os.path.join(EMBEDDINGS_DIR, filename)
    
    torch.save(dataset, output_path)
    
    print(f"\n  {modality_name} embeddings saved to: {output_path}")
    print(f"  File structure:")
    for person, tensor in dataset.items():
        size_mb = tensor.nelement() * tensor.element_size() / (1024 * 1024)
        print(f"    '{person}': Tensor{list(tensor.shape)} ({size_mb:.2f} MB)")


# ============================================================
#  MAIN — Run both pipelines
# ============================================================

def run_embedding_pipeline():
    """
    Main function — computes and saves embeddings for both faces and voices.
    """
    print("=" * 60)
    print("  EMBEDDING COMPUTATION PIPELINE")
    print("=" * 60)
    print(f"\n  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # ---- FACE EMBEDDINGS ----
    face_dataset = compute_face_embeddings()
    if face_dataset:
        save_embeddings(face_dataset, FACE_EMBEDDINGS_FILE, "Face")
    
    # ---- VOICE EMBEDDINGS ----
    voice_dataset = compute_voice_embeddings()
    if voice_dataset:
        save_embeddings(voice_dataset, VOICE_EMBEDDINGS_FILE, "Voice")
    
    # ---- FINAL SUMMARY ----
    print("\n" + "=" * 60)
    print("  EMBEDDING PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    
    if face_dataset:
        total_face = sum(t.shape[0] for t in face_dataset.values())
        print(f"\n  Face embeddings:  {total_face} vectors × 512 dimensions")
        for person, tensor in face_dataset.items():
            print(f"    {person}: {tensor.shape[0]} embeddings")
    
    if voice_dataset:
        total_voice = sum(t.shape[0] for t in voice_dataset.values())
        print(f"\n  Voice embeddings: {total_voice} vectors × 192 dimensions")
        for person, tensor in voice_dataset.items():
            print(f"    {person}: {tensor.shape[0]} embeddings")
    
    print(f"\n  Saved to: {EMBEDDINGS_DIR}/")
    print(f"    ├── {FACE_EMBEDDINGS_FILE}")
    print(f"    └── {VOICE_EMBEDDINGS_FILE}")
    
    print(f"\n  Next step: python data_preparation/enroll_users.py")
    print(f"  (Creates reference embeddings for each authorized person)\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    run_embedding_pipeline()