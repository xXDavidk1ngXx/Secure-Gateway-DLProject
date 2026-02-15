"""
Centralized Configuration File
================================
Single source of truth for ALL paths, thresholds, and hyperparameters
used across the entire project.

Why this file exists:
    Before this, every script had its own hardcoded values at the top.
    If you changed IMAGE_SIZE in one script but forgot another, things would
    break silently. Now every script imports from here, so you change a
    value once and it takes effect everywhere.

How to use in any script:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config import *

    Or more selectively:
    from utils.config import PATHS, FACE_SETTINGS, TRAINING

Organization:
    The config is split into logical sections, each as a simple Python class
    used like a namespace (e.g., PATHS.FACE_RAW, TRAINING.LEARNING_RATE).
    This keeps things organized without adding complexity — no YAML, no JSON,
    just plain Python that you can read and edit.
"""

import os
import torch


# ============================================================
#  PROJECT ROOT
# ============================================================
# All paths are relative to the project root directory.
# This auto-detects the root based on where utils/config.py lives.
# If config.py is at: /project/utils/config.py → ROOT = /project/

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  SECTION 1: FILE PATHS
# ============================================================
# Every input/output directory and file path in the project.
# Changing a folder name here updates it everywhere automatically.

class PATHS:
    """All file and directory paths used across the project."""

    # --- Raw data (input — your original photos and recordings) ---
    FACE_RAW        = os.path.join(PROJECT_ROOT, "data", "face", "raw")
    VOICE_RAW       = os.path.join(PROJECT_ROOT, "data", "voice", "raw")

    # --- Preprocessed data (cleaned, normalized, ready for augmentation) ---
    FACE_PROCESSED  = os.path.join(PROJECT_ROOT, "data", "face", "processed")
    VOICE_PROCESSED = os.path.join(PROJECT_ROOT, "data", "voice", "processed")

    # --- Augmented data (originals + augmented versions) ---
    FACE_AUGMENTED  = os.path.join(PROJECT_ROOT, "data", "face", "augmented")
    VOICE_AUGMENTED = os.path.join(PROJECT_ROOT, "data", "voice", "augmented")

    # --- Embeddings (numerical vectors from pretrained models) ---
    EMBEDDINGS_DIR      = os.path.join(PROJECT_ROOT, "data", "embeddings")
    FACE_EMBEDDINGS     = os.path.join(PROJECT_ROOT, "data", "embeddings", "face_embeddings.pt")
    VOICE_EMBEDDINGS    = os.path.join(PROJECT_ROOT, "data", "embeddings", "voice_embeddings.pt")

    # --- Enrollment (mean profiles for each user) ---
    ENROLLMENT_DIR      = os.path.join(PROJECT_ROOT, "data", "enrollment")

    # --- Models (trained model weights and user profiles) ---
    MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
    FUSION_MODEL        = os.path.join(PROJECT_ROOT, "models", "fusion_model.pt")
    USER_PROFILES       = os.path.join(PROJECT_ROOT, "models", "user_profiles.pt")

    # --- Data splits (saved train/val/test indices for reproducibility) ---
    DATA_SPLITS         = os.path.join(PROJECT_ROOT, "models", "data_splits.pt")

    # --- Password data ---
    PASSWORD_DIR        = os.path.join(PROJECT_ROOT, "data", "password")


# ============================================================
#  SECTION 2: FACE PREPROCESSING SETTINGS
# ============================================================
# Used by: preprocess_faces.py
# These control how MTCNN detects and crops faces from raw photos.

class FACE_PREPROCESS:
    """Settings for face detection, alignment, and cropping."""

    IMAGE_SIZE  = 160       # Output face image size in pixels (160x160 for FaceNet)
                            # FaceNet was trained on 160x160 — must match exactly

    MARGIN      = 40        # Extra pixels to include around the detected face bounding box
                            # Higher = more forehead/chin visible, but more background too
                            # 40 is the standard default for FaceNet

    MIN_FACE_SIZE = 50      # Minimum face size in pixels for MTCNN to detect
                            # Filters out tiny background faces
                            # Lower to 30 if your photos are low-resolution

    CONFIDENCE_THRESHOLD = 0.95     # Minimum MTCNN detection confidence (0 to 1)
                                    # 0.95 = strict, only keeps high-confidence detections
                                    # Lower to 0.90 if too many valid photos are rejected

    # MTCNN internal stage thresholds (P-Net, R-Net, O-Net)
    # These control how aggressive each stage is at filtering candidates
    # Default [0.6, 0.7, 0.7] works well — rarely needs changing
    MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]

    # Scale factor for the MTCNN image pyramid
    # Controls how much the image is downscaled at each step
    # 0.709 is the standard default
    MTCNN_SCALE_FACTOR = 0.709


# ============================================================
#  SECTION 3: VOICE PREPROCESSING SETTINGS
# ============================================================
# Used by: preprocess_voices.py
# These control how raw audio recordings are cleaned and normalized.

class VOICE_PREPROCESS:
    """Settings for audio resampling, trimming, and normalization."""

    SAMPLE_RATE     = 16000     # Target sample rate in Hz
                                # 16kHz is the standard for speech models
                                # (SpeechBrain, Whisper, Wav2Vec2, etc.)

    # --- Silence trimming ---
    TRIM_SILENCE        = True      # Whether to remove leading/trailing silence
    SILENCE_THRESHOLD_DB = -40      # Audio below this dB level = "silence"
                                    # Lower (e.g., -50) = less aggressive trimming
    MIN_SILENCE_DURATION = 0.1      # Minimum silence length (seconds) to trigger trim
    SILENCE_PADDING     = 0.15      # Seconds of silence to keep at start/end after trim
                                    # Avoids cutting off the first syllable

    # --- Volume normalization ---
    NORMALIZE_VOLUME    = True      # Whether to normalize audio amplitude
    PEAK_NORMALIZE_DB   = -1.0      # Target peak amplitude in dB
                                    # -1.0 dB = nearly max volume without clipping

    # --- Duration validation ---
    MIN_DURATION_SEC    = 0.5       # Reject recordings shorter than this (likely noise)
    MAX_DURATION_SEC    = 30.0      # Reject recordings longer than this (likely error)


# ============================================================
#  SECTION 4: FACE AUGMENTATION SETTINGS
# ============================================================
# Used by: augment_faces.py
# Controls how many and what kind of augmented face images are generated.

class FACE_AUGMENTATION:
    """Settings for face image augmentation."""

    AUGMENTATIONS_PER_IMAGE = 15    # Augmented versions generated per original image
                                    # With 30 originals: 30 + (30 × 15) = 480 images/person
                                    # Good balance between variety and disk space

    COPY_ORIGINALS = True           # Also copy the clean original images to the augmented folder
                                    # Makes the augmented folder self-contained

    IMAGE_SIZE = FACE_PREPROCESS.IMAGE_SIZE     # Must match preprocessing output (160)
                                                # Linked to FACE_PREPROCESS so they can't diverge

    RANDOM_SEED = 42                # For reproducibility — same seed = same augmentations
                                    # Set to None for different results each run


# ============================================================
#  SECTION 5: VOICE AUGMENTATION SETTINGS
# ============================================================
# Used by: augment_voices.py
# Controls how many augmented voice clips are generated.

class VOICE_AUGMENTATION:
    """Settings for voice audio augmentation."""

    AUGMENTATIONS_PER_CLIP = 3      # Augmented versions per original clip
                                    # With 30 originals: 30 + (30 × 3) = 120 clips/person
                                    # Fewer than face because:
                                    #   1. ECAPA-TDNN is already robust
                                    #   2. Heavy audio distortion destroys identity
                                    #   3. Each clip has thousands of audio frames

    COPY_ORIGINALS = True           # Copy original clean audio to augmented folder

    SAMPLE_RATE = VOICE_PREPROCESS.SAMPLE_RATE  # Must match preprocessing output (16kHz)

    RANDOM_SEED = 42                # For reproducibility


# ============================================================
#  SECTION 6: EMBEDDING COMPUTATION SETTINGS
# ============================================================
# Used by: compute_embeddings.py
# Controls how pretrained models convert images/audio to vectors.

class EMBEDDINGS:
    """Settings for embedding computation."""

    BATCH_SIZE  = 32        # Images/clips processed per batch
                            # 32 works for most setups with 8GB+ RAM
                            # Lower to 16 if you run out of memory

    # Embedding dimensions (fixed by the pretrained models — DO NOT change)
    FACE_EMBEDDING_DIM  = 512       # InceptionResnetV1 (FaceNet) output size
    VOICE_EMBEDDING_DIM = 192       # ECAPA-TDNN (SpeechBrain) output size
    FUSED_EMBEDDING_DIM = 512 + 192  # = 704, concatenated input to fusion model

    # Pretrained model identifiers
    FACE_MODEL_NAME  = "vggface2"   # FaceNet pretrained weights (alternatives: "casia-webface")
    VOICE_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"  # SpeechBrain model hub ID


# ============================================================
#  SECTION 7: ENROLLMENT SETTINGS
# ============================================================
# Used by: enroll_users.py
# Controls how user identity profiles are computed from embeddings.

class ENROLLMENT:
    """Settings for user profile generation."""

    # Method for computing the reference profile from multiple embeddings
    # "mean" = average all embeddings (simple, effective, noise-resistant)
    # Could potentially support "median" or "trimmed_mean" in the future
    AGGREGATION_METHOD = "mean"


# ============================================================
#  SECTION 8: TRAINING SETTINGS (FUSION MODEL)
# ============================================================
# Used by: train_model.py
# All hyperparameters for training the Late Fusion MLP.

class TRAINING:
    """Hyperparameters for fusion model training."""

    # --- Data split ratios ---
    # Must sum to 1.0
    # Split is done PER PERSON before pairing to prevent data leakage
    TRAIN_RATIO = 0.70      # 70% of embeddings used for training
    VAL_RATIO   = 0.15      # 15% for validation (monitor overfitting during training)
    TEST_RATIO  = 0.15      # 15% held out for final evaluation (used only once)

    # --- Model architecture ---
    HIDDEN_1    = 256       # First hidden layer size  (704 → 256)
    HIDDEN_2    = 128       # Second hidden layer size (256 → 128)
    DROPOUT_1   = 0.3       # Dropout rate after first hidden layer
    DROPOUT_2   = 0.2       # Dropout rate after second hidden layer

    # --- Optimizer ---
    LEARNING_RATE   = 1e-3      # Adam optimizer learning rate
    WEIGHT_DECAY    = 1e-4      # L2 regularization strength (prevents overfitting)
    EPOCHS          = 100       # Maximum training epochs
    BATCH_SIZE      = 64        # Training batch size

    # --- Learning rate scheduler ---
    # Reduces learning rate when validation loss stops improving
    SCHEDULER_PATIENCE  = 10    # Epochs to wait before reducing LR
    SCHEDULER_FACTOR    = 0.5   # Multiply LR by this factor when reducing

    # --- Early stopping ---
    # Stops training if validation loss doesn't improve for N epochs
    EARLY_STOPPING_PATIENCE = 20

    # --- Reproducibility ---
    RANDOM_SEED = 42


# ============================================================
#  SECTION 9: CLASS LABELS
# ============================================================
# The authorized users + the unknown/impostor class.
# Used by: train_model.py, run_system.py, smart_finetune.py

class CLASSES:
    """Class labels for the fusion model."""

    # Authorized user names — these MUST match the folder names in your data directories
    # Order matters: index 0 = "david", index 1 = "itzhak", index 2 = "yossi"
    AUTHORIZED_USERS = ["david", "itzhak", "yossi"]

    # The unknown/impostor class label
    UNKNOWN_LABEL = "unknown"

    # Complete class list (authorized + unknown)
    # The model's output neurons correspond to this list in order
    ALL_CLASSES = AUTHORIZED_USERS + [UNKNOWN_LABEL]

    # Number of output neurons
    NUM_CLASSES = len(ALL_CLASSES)

    # Mapping: class name → index and index → class name
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(ALL_CLASSES)}
    IDX_TO_CLASS = {idx: name for idx, name in enumerate(ALL_CLASSES)}


# ============================================================
#  SECTION 10: SYSTEM DECISION THRESHOLDS
# ============================================================
# Used by: run_system.py
# These control when the system grants/denies access.

class THRESHOLDS:
    """Decision thresholds for the live system."""

    # --- Model confidence thresholds ---
    HIGH_CONFIDENCE     = 0.85      # Above this → grant access immediately
                                    # Starting value; may be updated after Phase 5
                                    # based on Equal Error Rate (EER) analysis

    LOW_CONFIDENCE      = 0.50      # Below this → reject immediately (likely impostor)

    # Between LOW and HIGH → "gray area" → fallback to cosine similarity check

    # --- Cosine similarity thresholds (for gray area fallback) ---
    # Compared against enrollment profiles (mean embeddings)
    FACE_SIMILARITY_MIN = 0.4       # Minimum face similarity to accept
    VOICE_SIMILARITY_MIN = 0.4      # Minimum voice similarity to accept
    # Both must pass for gray area access to be granted

    # --- Password verification ---
    PASSWORD_FUZZY_THRESHOLD = 0.75     # Minimum SequenceMatcher ratio for password match
                                        # 0.75 = 75% similarity (tolerates minor transcription errors)

    PASSWORD_KEYWORDS = ["voice", "password"]   # Backup: if these words appear in transcription,
                                                # accept even if full fuzzy match fails


# ============================================================
#  SECTION 11: LIVE SYSTEM SETTINGS
# ============================================================
# Used by: run_system.py
# Everything needed for the live authentication pipeline:
# camera, microphone, password, audio cleanup, and retry logic.

class LIVE_SYSTEM:
    """Configuration for the live biometric authentication system."""

    # --- Camera ---
    CAMERA_INDEX = 0                # Webcam device index for OpenCV
                                    # 0 = built-in laptop webcam (most common)
                                    # 1 = external USB webcam
                                    # Change this if the wrong camera opens

    CAMERA_WARMUP_SEC = 1.0         # Seconds to wait after opening the camera
                                    # Some webcams need time to adjust exposure/white balance
                                    # Without this, the first frames may be black or washed out

    # --- Microphone recording ---
    RECORDING_DURATION_SEC = 6      # How long to record when user speaks (seconds)
                                    # 6 seconds gives enough time to say "my voice is my password"
                                    # comfortably, with margin for hesitation

    RECORDING_SAMPLE_RATE = VOICE_PREPROCESS.SAMPLE_RATE    # 16000 Hz — must match ECAPA-TDNN
    RECORDING_CHANNELS = 1          # Mono audio (1 channel) — speech models expect mono

    # --- Password ---
    PASSWORD_PHRASE = "my voice is my password"     # The passphrase all users must say
                                                    # Compared via fuzzy matching (not exact)
                                                    # Same phrase for all users — the VOICE
                                                    # is what identifies who they are,
                                                    # the WORDS are just the "something you know"

    # --- Live audio cleanup ---
    # The raw microphone recording is cleaned before use, applying the
    # same processing pipeline from preprocess_voices.py:
    #   1. Trim leading/trailing silence (removes dead air before/after speech)
    #   2. Normalize volume (ensures consistent amplitude for the model)
    # These are linked to VOICE_PREPROCESS so they stay consistent with training data.
    LIVE_TRIM_SILENCE       = VOICE_PREPROCESS.TRIM_SILENCE
    LIVE_SILENCE_THRESHOLD  = VOICE_PREPROCESS.SILENCE_THRESHOLD_DB
    LIVE_SILENCE_PADDING    = VOICE_PREPROCESS.SILENCE_PADDING
    LIVE_NORMALIZE          = VOICE_PREPROCESS.NORMALIZE_VOLUME
    LIVE_PEAK_DB            = VOICE_PREPROCESS.PEAK_NORMALIZE_DB

    # --- Live face detection ---
    # For real-time use, we slightly relax the MTCNN confidence threshold
    # compared to preprocessing. During preprocessing we could discard bad photos
    # and retake them — live, the user is standing there and we need to work
    # with whatever frame we get. 0.90 instead of 0.95.
    LIVE_FACE_CONFIDENCE = 0.90

    # --- Password recording ---
    PASSWORD_RECORD_DURATION = 5    # Seconds to record for the password phrase
                                    # Slightly shorter than RECORDING_DURATION_SEC since
                                    # the password is a known, short phrase

    # --- Face detection (live) ---
    FACE_DETECTION_TIMEOUT = 15     # Seconds to wait for a face before giving up
                                    # 15s gives the user time to position themselves

    MAX_FACE_RETRIES = 3            # How many times to retry face detection
                                    # Each retry gets a fresh FACE_DETECTION_TIMEOUT window

    # --- Authentication flow ---
    MAX_ATTEMPTS = 3                # How many password tries before the system locks out
                                    # After 3 failed attempts: "Access denied. Contact admin."
                                    # Prevents brute-force repeated attempts

    LOCKOUT_DURATION_SEC = 30       # Seconds to wait after max failed attempts
                                    # During lockout, the system refuses all attempts

    # --- Admin mode ---
    ADMIN_KEY = "a"                 # Key to press for admin override after a rejection
                                    # Admin can correct false rejections for Active Learning

    # --- Logging ---
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")    # Directory for authentication logs
    LOG_ATTEMPTS = True             # Whether to save every attempt to a log file
                                    # Logs include: timestamp, decision, confidence, transcript
                                    # Used for security auditing and Active Learning (Phase 4)


# ============================================================
#  SECTION 12: CAPTURE & PREVIEW SETTINGS
# ============================================================
# Used by: run_system.py (CaptureSession class)
# Controls multi-frame face capture, quality scoring, and live preview.

class CAPTURE:
    """Settings for multi-frame face capture and live preview window."""

    # --- Multi-frame face capture ---
    # Instead of using the first face found, the system records for
    # FACE_CAPTURE_DURATION seconds after the first detection, scores
    # every frame, and picks the best one for embedding.
    FACE_CAPTURE_DURATION = 3       # Seconds of video to collect after first face found
                                    # 3 seconds at 30fps = ~90 frames to choose from
                                    # Higher = more chances for a great frame, slower UX

    FACE_TOP_K_FRAMES = 3           # How many top-scoring frames to try embedding
                                    # If #1 fails MTCNN alignment, tries #2, then #3

    FACE_MIN_QUALITY = 0.3          # Minimum quality score to keep a frame as candidate
                                    # Frames below this are discarded immediately
                                    # Keeps memory usage reasonable during capture

    # --- Quality scoring weights (must sum to 1.0) ---
    # Each captured face frame is scored on three criteria:
    QUALITY_WEIGHT_CONFIDENCE = 0.4     # MTCNN detection confidence (higher = clearer face)
    QUALITY_WEIGHT_SIZE       = 0.3     # Face size relative to frame (bigger = closer = better)
    QUALITY_WEIGHT_CENTER     = 0.3     # How centered the face is (center = proper position)

    # --- Live preview window ---
    SHOW_PREVIEW = True                 # Enable/disable the OpenCV preview window
                                        # False = terminal-only mode (original behavior)
                                        # True = camera window with bounding boxes + voice bar

    PREVIEW_WINDOW_NAME = "Secure Gateway - Live Preview"

    # Bounding box colors (BGR format for OpenCV)
    BOX_COLOR_GOOD      = (0, 255, 0)       # Green  — good quality, frame is being recorded
    BOX_COLOR_UNCERTAIN = (0, 165, 255)     # Orange — face detected but low quality
    BOX_COLOR_NONE      = (0, 0, 255)       # Red    — no face detected
    BOX_THICKNESS       = 2                 # Line thickness for bounding boxes

    # --- Voice amplitude bar ---
    VOICE_BAR_HEIGHT    = 30                # Pixels tall for the amplitude bar
    VOICE_BAR_COLOR     = (0, 200, 0)       # Green fill for the bar
    VOICE_BAR_BG_COLOR  = (40, 40, 40)      # Dark gray background


# ============================================================
#  SECTION 13: DEVICE CONFIGURATION
# ============================================================
# Automatically selects GPU if available, otherwise CPU.

class DEVICE:
    """Compute device settings."""

    # Auto-detect: use GPU if available, else CPU
    COMPUTE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # For display purposes
    NAME = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"


# ============================================================
#  SECTION 14: ACTIVE LEARNING SETTINGS
# ============================================================
# Used by: smart_finetune.py
# Controls rapid retraining when admin corrects a false rejection.

class ACTIVE_LEARNING:
    """Settings for admin-mode rapid retraining."""

    FINETUNE_EPOCHS     = 5         # Epochs for rapid retraining (fast, ~3 seconds)
    FINETUNE_LR         = 5e-4      # Lower learning rate than initial training
                                    # to avoid destroying what the model already learned


# ============================================================
#  SECTION 15: LIVENESS DETECTION SETTINGS
# ============================================================
# Used by: run_system.py (check_liveness function)
# Controls anti-spoofing checks that run AFTER face capture
# and BEFORE the fusion model decision.
#
# Two complementary checks:
#   A. Blink Detection — tracks Eye Aspect Ratio (EAR) across frames.
#      A photo never blinks; a real person blinks ~1-2x in 3 seconds.
#   B. Head Pose Variation — measures micro-movements in yaw/pitch.
#      A photo has zero variation; a real person always moves slightly.
#
# Decision: pass if EITHER check passes (OR logic).
# This handles sunglasses (blink fails, pose works) and very still
# people (pose borderline, but they blink).
#
# Fail-safe: if not enough frames have usable landmarks, ACCESS DENIED.
# The system cannot verify liveness → refuses to proceed.

class LIVENESS:
    """Settings for post-capture liveness detection (anti-spoofing)."""

    ENABLED = True                      # Master toggle — False skips liveness entirely
                                        # Useful for demos, debugging, or when not needed

    # --- Blink Detection (Eye Aspect Ratio) ---
    # EAR = (dist(p2,p6) + dist(p3,p5)) / (2 * dist(p1,p4))
    # where p1-p6 are the 6 eye landmarks from MediaPipe Face Mesh.
    # When the eye is open, EAR ≈ 0.25-0.35.
    # When the eye closes (blink), EAR drops below ~0.20 for 2-3 frames.

    EAR_THRESHOLD = 0.22                # EAR below this = eye considered closed
                                        # 0.22 is slightly generous (catches partial blinks)
                                        # Standard in literature is 0.20-0.25

    BLINK_CONSEC_FRAMES = 2             # Consecutive low-EAR frames to count as 1 blink
                                        # 2 frames at 30fps = ~67ms — matches real blink duration
                                        # Lower to 1 for maximum sensitivity (may false-trigger)

    MIN_BLINKS_REQUIRED = 1             # Minimum blinks to pass the blink check
                                        # 1 is sufficient — average person blinks 15-20x per minute
                                        # so 1 in 3 seconds is very achievable

    EAR_VARIATION_THRESHOLD = 0.003     # Minimum std_dev of EAR across all frames to pass
                                        # Even without a full blink, real eyes have micro-fluctuations
                                        # A photo's EAR is perfectly constant (std ≈ 0)
                                        # 0.003 is very sensitive — catches the subtlest eye movement
                                        # This acts as a secondary blink signal

    # --- Head Pose Variation ---
    # Uses OpenCV solvePnP with 6 facial landmarks from MediaPipe
    # to estimate yaw (left-right) and pitch (up-down) per frame.
    # A real person has involuntary micro-movements even when "holding still."

    HEAD_POSE_MIN_STD_YAW = 0.3         # Minimum std_dev of yaw angle (degrees)
                                        # 0.3° is extremely sensitive — catches micro-sway
                                        # A printed photo will be ~0.0-0.1° (camera noise only)
                                        # A real person is typically 0.5-3.0° even when still

    HEAD_POSE_MIN_STD_PITCH = 0.3       # Minimum std_dev of pitch angle (degrees)
                                        # Same sensitivity as yaw — catches micro-nods

    # --- Decision Logic ---
    REQUIRE_ALL = False                 # False = pass if EITHER blink OR pose passes (recommended)
                                        # True = require BOTH (stricter — may reject with sunglasses)
                                        # OR logic ensures a real person always gets through

    # --- Frame Requirements ---
    MIN_FRAMES_FOR_ANALYSIS = 10        # Need at least this many frames with valid landmarks
                                        # Below this → FAIL-SAFE: deny access (can't verify)
                                        # 10 frames = ~0.33s of usable video at 30fps
                                        # Should always be met with 3s capture duration

    # --- MediaPipe Settings ---
    FACE_MESH_MAX_FACES = 1             # Only track one face (the person at the camera)
    FACE_MESH_DETECTION_CONF = 0.5      # Min detection confidence for face landmark detection
    FACE_MESH_PRESENCE_CONF = 0.5       # Min face presence confidence
    FACE_MESH_TRACKING_CONF = 0.5       # Min tracking confidence between frames

    # MediaPipe FaceLandmarker model file (.task bundle)
    # Downloaded automatically on first run if not present.
    # Stored alongside other models in the models/ directory.
    FACE_LANDMARKER_MODEL = "face_landmarker.task"
    FACE_LANDMARKER_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    # --- MediaPipe Eye Landmark Indices (468-point face mesh) ---
    # 6 points per eye used for EAR computation
    LEFT_EYE = [33, 160, 158, 133, 153, 144]       # outer, upper2, upper1, inner, lower1, lower2
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]     # outer, upper2, upper1, inner, lower1, lower2

    # --- Head Pose Landmark Indices ---
    # 6 facial landmarks used for solvePnP head pose estimation
    POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]    # nose tip, chin, L eye, R eye, L mouth, R mouth


# ============================================================
#  HELPER: Print current configuration
# ============================================================

def print_config():
    """Print a summary of the current configuration. Useful for debugging."""

    print("=" * 60)
    print("  CURRENT CONFIGURATION")
    print("=" * 60)

    print(f"\n  Project root: {PROJECT_ROOT}")
    print(f"  Device:       {DEVICE.NAME} ({DEVICE.COMPUTE})")

    print(f"\n  --- Paths ---")
    print(f"  Face raw:         {PATHS.FACE_RAW}")
    print(f"  Face processed:   {PATHS.FACE_PROCESSED}")
    print(f"  Face augmented:   {PATHS.FACE_AUGMENTED}")
    print(f"  Voice raw:        {PATHS.VOICE_RAW}")
    print(f"  Voice processed:  {PATHS.VOICE_PROCESSED}")
    print(f"  Voice augmented:  {PATHS.VOICE_AUGMENTED}")
    print(f"  Embeddings:       {PATHS.EMBEDDINGS_DIR}")
    print(f"  Models:           {PATHS.MODELS_DIR}")

    print(f"\n  --- Face Preprocessing ---")
    print(f"  Image size:    {FACE_PREPROCESS.IMAGE_SIZE}x{FACE_PREPROCESS.IMAGE_SIZE}")
    print(f"  Margin:        {FACE_PREPROCESS.MARGIN}px")
    print(f"  Min face size: {FACE_PREPROCESS.MIN_FACE_SIZE}px")
    print(f"  Confidence:    {FACE_PREPROCESS.CONFIDENCE_THRESHOLD}")

    print(f"\n  --- Voice Preprocessing ---")
    print(f"  Sample rate:   {VOICE_PREPROCESS.SAMPLE_RATE} Hz")
    print(f"  Trim silence:  {VOICE_PREPROCESS.TRIM_SILENCE}")
    print(f"  Normalize:     {VOICE_PREPROCESS.NORMALIZE_VOLUME}")
    print(f"  Duration:      {VOICE_PREPROCESS.MIN_DURATION_SEC}s – {VOICE_PREPROCESS.MAX_DURATION_SEC}s")

    print(f"\n  --- Augmentation ---")
    print(f"  Face:  {FACE_AUGMENTATION.AUGMENTATIONS_PER_IMAGE} per image (seed={FACE_AUGMENTATION.RANDOM_SEED})")
    print(f"  Voice: {VOICE_AUGMENTATION.AUGMENTATIONS_PER_CLIP} per clip  (seed={VOICE_AUGMENTATION.RANDOM_SEED})")

    print(f"\n  --- Embeddings ---")
    print(f"  Face dim:   {EMBEDDINGS.FACE_EMBEDDING_DIM}")
    print(f"  Voice dim:  {EMBEDDINGS.VOICE_EMBEDDING_DIM}")
    print(f"  Fused dim:  {EMBEDDINGS.FUSED_EMBEDDING_DIM}")
    print(f"  Batch size: {EMBEDDINGS.BATCH_SIZE}")

    print(f"\n  --- Training ---")
    print(f"  Split:         {TRAINING.TRAIN_RATIO}/{TRAINING.VAL_RATIO}/{TRAINING.TEST_RATIO} (train/val/test)")
    print(f"  Architecture:  {EMBEDDINGS.FUSED_EMBEDDING_DIM} → {TRAINING.HIDDEN_1} → {TRAINING.HIDDEN_2} → {CLASSES.NUM_CLASSES}")
    print(f"  Dropout:       {TRAINING.DROPOUT_1}/{TRAINING.DROPOUT_2}")
    print(f"  LR:            {TRAINING.LEARNING_RATE} (decay={TRAINING.WEIGHT_DECAY})")
    print(f"  Epochs:        {TRAINING.EPOCHS} (early stop={TRAINING.EARLY_STOPPING_PATIENCE})")
    print(f"  Batch size:    {TRAINING.BATCH_SIZE}")

    print(f"\n  --- Classes ---")
    print(f"  Authorized:    {CLASSES.AUTHORIZED_USERS}")
    print(f"  Total classes: {CLASSES.NUM_CLASSES} ({CLASSES.ALL_CLASSES})")

    print(f"\n  --- Thresholds ---")
    print(f"  High confidence:   {THRESHOLDS.HIGH_CONFIDENCE}")
    print(f"  Low confidence:    {THRESHOLDS.LOW_CONFIDENCE}")
    print(f"  Face similarity:   {THRESHOLDS.FACE_SIMILARITY_MIN}")
    print(f"  Voice similarity:  {THRESHOLDS.VOICE_SIMILARITY_MIN}")
    print(f"  Password fuzzy:    {THRESHOLDS.PASSWORD_FUZZY_THRESHOLD}")

    print(f"\n  --- Live System ---")
    print(f"  Camera index:      {LIVE_SYSTEM.CAMERA_INDEX}")
    print(f"  Recording:         {LIVE_SYSTEM.RECORDING_DURATION_SEC}s @ "
          f"{LIVE_SYSTEM.RECORDING_SAMPLE_RATE}Hz (mono)")
    print(f"  Password phrase:   \"{LIVE_SYSTEM.PASSWORD_PHRASE}\"")
    print(f"  Password record:   {LIVE_SYSTEM.PASSWORD_RECORD_DURATION}s")
    print(f"  Face confidence:   {LIVE_SYSTEM.LIVE_FACE_CONFIDENCE} (live, relaxed)")
    print(f"  Face timeout:      {LIVE_SYSTEM.FACE_DETECTION_TIMEOUT}s")
    print(f"  Face retries:      {LIVE_SYSTEM.MAX_FACE_RETRIES}")
    print(f"  Max attempts:      {LIVE_SYSTEM.MAX_ATTEMPTS} "
          f"(lockout={LIVE_SYSTEM.LOCKOUT_DURATION_SEC}s)")
    print(f"  Admin key:         '{LIVE_SYSTEM.ADMIN_KEY}'")
    print(f"  Log attempts:      {LIVE_SYSTEM.LOG_ATTEMPTS} → {LIVE_SYSTEM.LOG_DIR}")

    print(f"\n  --- Capture & Preview ---")
    print(f"  Show preview:      {CAPTURE.SHOW_PREVIEW}")
    print(f"  Capture duration:  {CAPTURE.FACE_CAPTURE_DURATION}s after first face")
    print(f"  Top-K frames:      {CAPTURE.FACE_TOP_K_FRAMES}")
    print(f"  Min quality:       {CAPTURE.FACE_MIN_QUALITY}")
    print(f"  Quality weights:   conf={CAPTURE.QUALITY_WEIGHT_CONFIDENCE}, "
          f"size={CAPTURE.QUALITY_WEIGHT_SIZE}, "
          f"center={CAPTURE.QUALITY_WEIGHT_CENTER}")

    print(f"\n  --- Liveness Detection ---")
    print(f"  Enabled:           {LIVENESS.ENABLED}")
    if LIVENESS.ENABLED:
        print(f"  EAR threshold:     {LIVENESS.EAR_THRESHOLD}")
        print(f"  Min blinks:        {LIVENESS.MIN_BLINKS_REQUIRED}")
        print(f"  EAR variation:     {LIVENESS.EAR_VARIATION_THRESHOLD}")
        print(f"  Head pose min std: yaw={LIVENESS.HEAD_POSE_MIN_STD_YAW}°, "
              f"pitch={LIVENESS.HEAD_POSE_MIN_STD_PITCH}°")
        print(f"  Logic:             {'AND (both required)' if LIVENESS.REQUIRE_ALL else 'OR (either passes)'}")
        print(f"  Min frames:        {LIVENESS.MIN_FRAMES_FOR_ANALYSIS}")
        print(f"  Fail-safe:         deny if insufficient frames")

    print()


# ============================================================
#  RUN — Print config when executed directly
# ============================================================
# Useful for verifying your configuration:
#   python utils/config.py

if __name__ == "__main__":
    print_config()
