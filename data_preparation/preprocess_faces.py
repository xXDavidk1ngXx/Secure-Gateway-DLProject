"""
Face Data Preprocessing Pipeline
=================================
Takes raw photos from data/face/raw/<person_name>/
Detects faces, aligns, crops, and resizes to 160x160
Saves clean face images to data/face/processed/<person_name>/

Usage:
    python data_preparation/preprocess_faces.py

Requirements:
    pip install facenet-pytorch Pillow opencv-python tqdm
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm


# ============================================================
#  CONFIGURATION — adjust these to fit your setup
# ============================================================

# Paths (relative to project root)
RAW_DATA_DIR = "data/face/raw"              # Input: raw photos organized by person
PROCESSED_DATA_DIR = "data/face/processed"  # Output: clean 160x160 face images

# Face detection & cropping settings
IMAGE_SIZE = 160        # Output face image size (160x160 for FaceNet)
MARGIN = 40             # Extra pixels around the detected face (adds context)
                        # Higher margin = more forehead/chin visible
                        # 40 is a good default, increase to 60 if faces look too tight

MIN_FACE_SIZE = 50      # Minimum face size in pixels to detect
                        # Filters out tiny faces in the background
                        # Lower this if your photos are low resolution

CONFIDENCE_THRESHOLD = 0.95  # Minimum detection confidence (0 to 1)
                             # MTCNN outputs a probability that the detection is a real face
                             # 0.95 is strict — only keeps high-confidence detections
                             # Lower to 0.90 if you're losing too many valid photos


# ============================================================
#  STEP 1: Initialize the face detector
# ============================================================

def create_detector(device="cpu"):
    """
    Create an MTCNN face detector.
    
    MTCNN (Multi-Task Cascaded Convolutional Network) works in 3 stages:
      Stage 1 (P-Net): Scans the image at multiple scales to find candidate face regions
      Stage 2 (R-Net): Refines the candidates, removes false positives
      Stage 3 (O-Net): Final refinement + outputs 5 facial landmarks
    
    The landmarks (left_eye, right_eye, nose, mouth_left, mouth_right) are used
    for alignment — rotating the face so the eyes are horizontal.
    """
    detector = MTCNN(
        image_size=IMAGE_SIZE,      # Output size after crop
        margin=MARGIN,              # Pixels to add around the face
        min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7],  # Confidence thresholds for the 3 MTCNN stages
        factor=0.709,               # Scale factor for image pyramid (default)
        post_process=False,         # If True, normalizes pixel values (we'll do it ourselves later)
        keep_all=False,             # Only keep the largest/most confident face
        device=device,
        select_largest=True,        # If multiple faces, pick the largest one
    )
    return detector


# ============================================================
#  STEP 2: Detect and extract a face from a single image
# ============================================================

def detect_and_crop_face(detector, image_path):
    """
    Given a path to an image:
      1. Load the image
      2. Detect the face
      3. Align using eye landmarks (MTCNN does this internally)
      4. Crop to 160x160
      5. Return the cropped face as a PIL Image
    
    Returns:
        face_image (PIL.Image): The cropped, aligned face (160x160 RGB)
        confidence (float): Detection confidence score
        OR
        None, None if no face was detected
    """
    try:
        # Load the image as RGB (MTCNN expects RGB, not BGR)
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Could not load image: {image_path} — {e}")
        return None, None
    
    # Detect face — this does detection + alignment + crop in one call
    # face_tensor: a PyTorch tensor of shape (3, 160, 160) with pixel values 0-255
    # confidence: float between 0 and 1
    face_tensor, confidence = detector.detect(img)
    
    # Check if a face was found
    if face_tensor is None or len(face_tensor) == 0:
        return None, None
    
    # confidence is an array of confidences for each detected face
    # Since keep_all=False, we get the best one, but detect() returns arrays
    best_confidence = confidence[0]
    
    if best_confidence < CONFIDENCE_THRESHOLD:
        return None, best_confidence
    
    # Now use MTCNN's __call__ method which does detect + align + crop + resize
    # This is the method that actually returns the cropped face
    face_cropped = detector(img)  # Returns tensor of shape (3, 160, 160) or None
    
    if face_cropped is None:
        return None, None
    
    # Convert tensor to PIL Image for saving
    # The tensor has values in [0, 255] range (since post_process=False)
    face_np = face_cropped.permute(1, 2, 0).numpy()  # (3,160,160) → (160,160,3)
    
    # Clip values to valid range and convert to uint8
    face_np = np.clip(face_np, 0, 255).astype(np.uint8)
    face_image = Image.fromarray(face_np)
    
    return face_image, best_confidence


# ============================================================
#  STEP 3: Process all images for a single person
# ============================================================

def process_person(detector, person_name, raw_dir, output_dir):
    """
    Process all images for one person:
      - Reads from: raw_dir/<person_name>/
      - Saves to:   output_dir/<person_name>/
    
    Returns a summary dict with counts of success/failure.
    """
    input_folder = os.path.join(raw_dir, person_name)
    output_folder = os.path.join(output_dir, person_name)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    
    # Get all image files
    image_files = [
        f for f in sorted(os.listdir(input_folder))
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"  [WARNING] No images found in {input_folder}")
        return {"total": 0, "success": 0, "failed": 0, "low_confidence": 0}
    
    stats = {"total": len(image_files), "success": 0, "failed": 0, "low_confidence": 0}
    failed_files = []
    
    print(f"\n  Processing {len(image_files)} images for '{person_name}'...")
    
    for filename in tqdm(image_files, desc=f"  {person_name}", leave=True):
        image_path = os.path.join(input_folder, filename)
        
        # Detect and crop the face
        face_image, confidence = detect_and_crop_face(detector, image_path)
        
        if face_image is None:
            if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
                stats["low_confidence"] += 1
                failed_files.append((filename, f"low confidence: {confidence:.3f}"))
            else:
                stats["failed"] += 1
                failed_files.append((filename, "no face detected"))
            continue
        
        # Save the processed face image
        # Use original filename but ensure .png extension for consistency
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)
        face_image.save(output_path, "PNG")
        
        stats["success"] += 1
    
    # Report results
    print(f"  Results for '{person_name}':")
    print(f"    ✓ Success:        {stats['success']}/{stats['total']}")
    print(f"    ✗ No face found:  {stats['failed']}")
    print(f"    ⚠ Low confidence: {stats['low_confidence']}")
    
    if failed_files:
        print(f"  Failed files:")
        for fname, reason in failed_files:
            print(f"    - {fname}: {reason}")
    
    return stats


# ============================================================
#  STEP 4: Run the full pipeline
# ============================================================

def run_preprocessing():
    """
    Main function — processes all people in the raw data directory.
    """
    print("=" * 60)
    print("  FACE DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Verify input directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"\n[ERROR] Raw data directory not found: {RAW_DATA_DIR}")
        print(f"Expected structure:")
        print(f"  {RAW_DATA_DIR}/")
        print(f"    ├── david/")
        print(f"    │   ├── photo1.jpg")
        print(f"    │   ├── photo2.jpg")
        print(f"    │   └── ...")
        print(f"    ├── itzhak/")
        print(f"    └── yossi/")
        sys.exit(1)
    
    # Find all person folders
    person_folders = [
        d for d in sorted(os.listdir(RAW_DATA_DIR))
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
        and not d.startswith(".")  # Skip hidden folders
    ]
    
    if not person_folders:
        print(f"\n[ERROR] No person folders found in {RAW_DATA_DIR}")
        sys.exit(1)
    
    print(f"\nFound {len(person_folders)} people: {', '.join(person_folders)}")
    print(f"Input:  {RAW_DATA_DIR}")
    print(f"Output: {PROCESSED_DATA_DIR}")
    print(f"Settings: image_size={IMAGE_SIZE}, margin={MARGIN}, min_face={MIN_FACE_SIZE}")
    
    # Create output directory
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Initialize the face detector
    print("\nLoading MTCNN face detector...")
    detector = create_detector(device="cpu")  # Change to "cuda" if you have a GPU
    print("Detector ready.\n")
    
    # Process each person
    all_stats = {}
    for person_name in person_folders:
        stats = process_person(detector, person_name, RAW_DATA_DIR, PROCESSED_DATA_DIR)
        all_stats[person_name] = stats
    
    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE — SUMMARY")
    print("=" * 60)
    
    total_images = sum(s["total"] for s in all_stats.values())
    total_success = sum(s["success"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    total_low_conf = sum(s["low_confidence"] for s in all_stats.values())
    
    print(f"\n  {'Person':<15} {'Total':<8} {'Success':<10} {'Failed':<10} {'Low Conf':<10}")
    print(f"  {'-'*53}")
    for person, stats in all_stats.items():
        print(f"  {person:<15} {stats['total']:<8} {stats['success']:<10} {stats['failed']:<10} {stats['low_confidence']:<10}")
    print(f"  {'-'*53}")
    print(f"  {'TOTAL':<15} {total_images:<8} {total_success:<10} {total_failed:<10} {total_low_conf:<10}")
    
    success_rate = (total_success / total_images * 100) if total_images > 0 else 0
    print(f"\n  Overall success rate: {success_rate:.1f}%")
    
    if total_failed > 0 or total_low_conf > 0:
        print(f"\n  Tips for failed images:")
        print(f"  - Make sure the face is clearly visible and well-lit")
        print(f"  - Avoid heavy occlusion (hands covering face, etc.)")
        print(f"  - If too many fail, try lowering CONFIDENCE_THRESHOLD to 0.90")
        print(f"  - If faces are small in the photo, try lowering MIN_FACE_SIZE to 30")
    
    print(f"\n  Processed images saved to: {PROCESSED_DATA_DIR}/")
    print(f"  You can now proceed to data augmentation.\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    run_preprocessing()
