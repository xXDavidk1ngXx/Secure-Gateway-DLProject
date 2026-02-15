"""
Face Data Augmentation Pipeline
=================================
Takes clean cropped faces from data/face/processed/<person_name>/
Generates N augmented versions of each image to simulate real-world variation.
Saves augmented images to data/face/augmented/<person_name>/

Why augment?
    You have ~30 photos per person. That's not enough for robust embeddings.
    Augmentation creates realistic variations that the system will encounter
    in production: different lighting, slight angles, partial occlusion, etc.
    After augmentation you'll have ~480 images per person (30 originals + 450 augmented).

Why save to disk instead of augmenting on-the-fly during embedding?
    1. You can visually inspect the augmented images before computing embeddings
    2. If you change embedding models later, you don't redo augmentation
    3. Reproducibility — saved images give the same embeddings every time
    4. Debugging — if embeddings are bad, you can check if augmentation is the cause

Expected input:
    data/face/processed/
    ├── david/
    │   ├── photo1.png
    │   ├── photo2.png
    │   └── ...
    ├── itzhak/
    └── yossi/

Output:
    data/face/augmented/
    ├── david/
    │   ├── photo1_original.png       ← copy of the clean original
    │   ├── photo1_aug_01.png         ← augmented version 1
    │   ├── photo1_aug_02.png         ← augmented version 2
    │   ├── ...
    │   ├── photo1_aug_15.png         ← augmented version 15
    │   ├── photo2_original.png
    │   └── ...
    ├── itzhak/
    └── yossi/

Usage:
    python data_preparation/augment_faces.py

Requirements:
    pip install Pillow torchvision tqdm numpy
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from tqdm import tqdm


# ============================================================
#  CONFIGURATION — adjust these to fit your setup
# ============================================================

# Paths (relative to project root)
PROCESSED_DATA_DIR = "data/face/processed"   # Input: clean 160x160 face images
AUGMENTED_DATA_DIR = "data/face/augmented"   # Output: original + augmented images

# Augmentation settings
AUGMENTATIONS_PER_IMAGE = 15    # Number of augmented versions per original image
                                 # With 30 originals: 30 + (30 × 15) = 480 images per person
                                 # This is a good balance — enough variety without being excessive

COPY_ORIGINALS = True            # Also copy the original clean images to the augmented folder
                                 # This way the augmented folder is self-contained:
                                 # the embedding script only needs to look in one place

IMAGE_SIZE = 160                 # Expected input size (should match preprocessing output)

RANDOM_SEED = 42                 # For reproducibility — same seed = same augmentations
                                 # Set to None for different results each run


# ============================================================
#  STEP 1: Define the augmentation transforms
# ============================================================
#
#  Each augmentation simulates a real-world condition your
#  security camera / webcam will encounter:
#
#  ┌────────────────────────┬──────────────────────────────────────┐
#  │ Augmentation           │ Real-world scenario                  │
#  ├────────────────────────┼──────────────────────────────────────┤
#  │ Horizontal flip        │ Person approaches from either side   │
#  │ Rotation ±15°          │ Slight head tilt                     │
#  │ Brightness jitter      │ Indoor vs outdoor, day vs night      │
#  │ Contrast jitter        │ Shadow on part of face               │
#  │ Saturation jitter      │ Different camera white balance       │
#  │ Gaussian blur          │ Slightly out-of-focus camera         │
#  │ Gaussian noise         │ Low-light camera sensor noise        │
#  │ Random erasing         │ Partial occlusion (glasses, mask)    │
#  │ Perspective transform  │ Camera at different angles/heights   │
#  └────────────────────────┴──────────────────────────────────────┘
#
#  IMPORTANT: We do NOT apply:
#  - Vertical flip (upside-down faces don't happen in real life)
#  - Heavy color changes (skin color should stay realistic)
#  - Large crops (face is already tightly cropped to 160x160)
#  - Extreme rotations (>20° would be unrealistic for a security setup)

def get_augmentation_transform():
    """
    Returns a single composed augmentation transform.
    
    Each call to this transform on an image produces a DIFFERENT
    random result because the transforms use random parameters.
    
    The transforms are applied in sequence:
      1. Geometric transforms (flip, rotate, perspective)
      2. Color transforms (brightness, contrast, saturation)
      3. Degradation transforms (blur, noise, erasing)
    
    Each transform has a probability (p) — not every transform
    fires every time. This creates diverse combinations.
    """
    transform = transforms.Compose([
        # --- Geometric transforms ---
        transforms.RandomHorizontalFlip(p=0.5),
        # 50% chance of mirror flip
        # Faces are roughly symmetric, so flipped faces are still valid

        transforms.RandomRotation(
            degrees=15,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,  # Fill exposed corners with black
        ),
        # Rotates ±15°. Simulates slight head tilt.
        # BILINEAR interpolation avoids jagged edges.
        # fill=0 (black) for exposed corners — these are small at 15°

        transforms.RandomPerspective(
            distortion_scale=0.1,   # Subtle perspective warp
            p=0.3,                  # Only 30% of the time
            fill=0,
        ),
        # Simulates camera viewing angle differences.
        # Low distortion_scale keeps the face recognizable.

        # --- Color transforms ---
        transforms.ColorJitter(
            brightness=0.3,    # ±30% brightness variation
            contrast=0.2,      # ±20% contrast variation
            saturation=0.2,    # ±20% color saturation
            hue=0.02,          # Very slight hue shift (keeps skin tones realistic)
        ),
        # This single transform handles most lighting variation.
        # The hue is kept very low — we don't want green-tinted faces.

        # --- Degradation transforms ---
        # Gaussian blur (simulates out-of-focus camera)
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,      # Small kernel for subtle blur
                sigma=(0.1, 1.5),   # Random blur strength
            ),
        ], p=0.3),  # Only 30% of the time

        # Random erasing (simulates partial occlusion)
        transforms.ToTensor(),  # Need tensor for RandomErasing
        transforms.RandomErasing(
            p=0.2,                  # 20% chance
            scale=(0.02, 0.08),     # Erased area is 2–8% of the image
            ratio=(0.3, 3.3),       # Rectangle aspect ratio
            value=0,                # Fill with black
        ),
        transforms.ToPILImage(),  # Back to PIL for saving
    ])
    return transform


def add_gaussian_noise(image, mean=0, std_range=(0.01, 0.03)):
    """
    Add random Gaussian noise to an image.
    
    This simulates camera sensor noise, especially in low-light conditions.
    torchvision doesn't have a built-in noise transform, so we do it manually.
    
    Args:
        image: PIL Image (RGB, 0-255)
        mean: Noise mean (0 = no bias)
        std_range: Range of noise standard deviation (relative to [0,1] scale)
    
    Returns:
        Noisy PIL Image
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.float32)
    
    noisy = np.clip(img_array + noise, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    
    return Image.fromarray(noisy)


def apply_single_augmentation(image, transform):
    """
    Apply augmentation to a single image with optional noise addition.
    
    We separate noise from the main transform pipeline because 
    numpy-based noise doesn't fit neatly into torchvision.Compose.
    
    Args:
        image: PIL Image (160x160 RGB)
        transform: The composed torchvision transform
    
    Returns:
        Augmented PIL Image (160x160 RGB)
    """
    # Apply the main augmentation pipeline
    augmented = transform(image)
    
    # 30% chance of adding Gaussian noise on top
    if random.random() < 0.3:
        augmented = add_gaussian_noise(augmented)
    
    # Ensure output size is correct (some transforms might change it slightly)
    if augmented.size != (IMAGE_SIZE, IMAGE_SIZE):
        augmented = augmented.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    return augmented


# ============================================================
#  STEP 2: Process all images for a single person
# ============================================================

def process_person(person_name, input_dir, output_dir):
    """
    Generate augmented images for one person.
    
    For each original image:
      1. Copy the original to the output folder (marked with _original suffix)
      2. Generate AUGMENTATIONS_PER_IMAGE augmented versions
    
    Returns:
        dict with counts: total originals, total augmented, any failures
    """
    input_folder = os.path.join(input_dir, person_name)
    output_folder = os.path.join(output_dir, person_name)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all processed face images
    valid_extensions = {".png", ".jpg", ".jpeg"}
    image_files = [
        f for f in sorted(os.listdir(input_folder))
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"  [WARNING] No images found in {input_folder}")
        return {"originals": 0, "augmented": 0, "failed": 0}
    
    stats = {"originals": 0, "augmented": 0, "failed": 0}
    
    # Create the augmentation transform
    transform = get_augmentation_transform()
    
    print(f"\n  Augmenting {len(image_files)} images for '{person_name}' "
          f"(×{AUGMENTATIONS_PER_IMAGE} each)...")
    
    for filename in tqdm(image_files, desc=f"  {person_name}", leave=True):
        image_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] Could not load {filename}: {e}")
            stats["failed"] += 1
            continue
        
        # --- Save the original (unmodified) ---
        if COPY_ORIGINALS:
            original_path = os.path.join(output_folder, f"{base_name}_original.png")
            img.save(original_path, "PNG")
            stats["originals"] += 1
        
        # --- Generate augmented versions ---
        for aug_idx in range(1, AUGMENTATIONS_PER_IMAGE + 1):
            try:
                augmented = apply_single_augmentation(img, transform)
                
                aug_filename = f"{base_name}_aug_{aug_idx:02d}.png"
                aug_path = os.path.join(output_folder, aug_filename)
                augmented.save(aug_path, "PNG")
                
                stats["augmented"] += 1
                
            except Exception as e:
                print(f"  [ERROR] Augmentation {aug_idx} failed for {filename}: {e}")
                stats["failed"] += 1
    
    total_generated = stats["originals"] + stats["augmented"]
    print(f"  Results for '{person_name}':")
    print(f"    ✓ Originals copied: {stats['originals']}")
    print(f"    ✓ Augmented:        {stats['augmented']}")
    print(f"    ✓ Total images:     {total_generated}")
    print(f"    ✗ Failed:           {stats['failed']}")
    
    return stats


# ============================================================
#  STEP 3: Run the full augmentation pipeline
# ============================================================

def run_augmentation():
    """
    Main function — augments all people in the processed data directory.
    """
    print("=" * 60)
    print("  FACE DATA AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Set random seed for reproducibility
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        print(f"\n  Random seed: {RANDOM_SEED} (results are reproducible)")
    else:
        print(f"\n  Random seed: None (results will vary each run)")
    
    # Verify input directory exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"\n[ERROR] Processed data directory not found: {PROCESSED_DATA_DIR}")
        print(f"Run preprocess_faces.py first!")
        sys.exit(1)
    
    # Find all person folders
    person_folders = [
        d for d in sorted(os.listdir(PROCESSED_DATA_DIR))
        if os.path.isdir(os.path.join(PROCESSED_DATA_DIR, d))
        and not d.startswith(".")
    ]
    
    if not person_folders:
        print(f"\n[ERROR] No person folders found in {PROCESSED_DATA_DIR}")
        sys.exit(1)
    
    print(f"\n  Found {len(person_folders)} people: {', '.join(person_folders)}")
    print(f"  Input:  {PROCESSED_DATA_DIR}")
    print(f"  Output: {AUGMENTED_DATA_DIR}")
    print(f"  Settings: {AUGMENTATIONS_PER_IMAGE} augmentations per image, "
          f"copy_originals={COPY_ORIGINALS}")
    
    # Create output directory
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
    
    # Process each person
    all_stats = {}
    for person_name in person_folders:
        stats = process_person(person_name, PROCESSED_DATA_DIR, AUGMENTED_DATA_DIR)
        all_stats[person_name] = stats
    
    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("  AUGMENTATION COMPLETE — SUMMARY")
    print("=" * 60)
    
    total_originals = sum(s["originals"] for s in all_stats.values())
    total_augmented = sum(s["augmented"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    total_images = total_originals + total_augmented
    
    print(f"\n  {'Person':<15} {'Originals':<12} {'Augmented':<12} {'Total':<10} {'Failed':<10}")
    print(f"  {'-' * 59}")
    for person, stats in all_stats.items():
        total = stats["originals"] + stats["augmented"]
        print(f"  {person:<15} {stats['originals']:<12} {stats['augmented']:<12} {total:<10} {stats['failed']:<10}")
    print(f"  {'-' * 59}")
    print(f"  {'TOTAL':<15} {total_originals:<12} {total_augmented:<12} {total_images:<10} {total_failed:<10}")
    
    print(f"\n  Augmented images saved to: {AUGMENTED_DATA_DIR}/")
    print(f"\n  Tip: Visually inspect some augmented images before proceeding.")
    print(f"  Look for images that are too distorted or unrecognizable.")
    print(f"  Delete any that look wrong — quality > quantity.")
    print(f"\n  Next step: python data_preparation/augment_voices.py")
    print(f"  Then:      python data_preparation/compute_embeddings.py\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    run_augmentation()