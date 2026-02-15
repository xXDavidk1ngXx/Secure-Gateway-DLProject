"""
=============================================================================
  SECURE GATEWAY — Multi-Modal Biometric Authentication System
  Main Executable (Phase 3)
=============================================================================

  This is the main entry point for the biometric security system.
  It orchestrates the full authentication pipeline:

      1. VOICE PASSWORD (Gate 1 — "Something you know")
         → Records the user speaking the password phrase
         → Google Speech-to-Text API transcribes the audio
         → Fuzzy matching checks if the transcription matches the stored password
         → If the password fails → REJECTED (no biometric check happens)

      2. BIOMETRIC SCAN (Gate 2 — "Something you are")
         → Captures a face frame from the webcam
         → Records a voice clip from the microphone
         → Computes face embedding (FaceNet / InceptionResnetV1 — 512-dim)
         → Computes voice embedding (ECAPA-TDNN / SpeechBrain — 192-dim)
         → Concatenates into a 704-dim fused vector

      3. FUSION MODEL DECISION (The Brain)
         → The Late Fusion MLP processes the 704-dim vector
         → Outputs probabilities for [David, Yossi, Itzhak, Unknown]
         → Applies 3-tier confidence logic:
            ┌─────────────────────────────────────────────────────┐
            │ Confidence ≥ 85%  →  ACCESS GRANTED immediately     │
            │ Confidence 50-85% →  Gray Area → Cosine fallback    │
            │ Confidence < 50%  →  ACCESS DENIED (impostor/noise) │
            └─────────────────────────────────────────────────────┘

      4. GRAY AREA FALLBACK (Safety Net)
         → Compares live embeddings against enrollment profiles (user_profiles.pt)
         → If face_similarity > 0.4 AND voice_similarity > 0.4 → GRANT
         → Otherwise → DENY

      5. ANTI-SPOOFING (Built-in)
         → Cross-modal mismatch (Face A + Voice B) triggers the Unknown neuron
         → The model was trained on cross-person pairs labeled as "unknown"
         → Mismatched face+voice → Unknown class activates → REJECTED

      6. LIVENESS DETECTION (Anti-Spoofing Gate — Phase 6)
         → Runs AFTER face capture, BEFORE fusion model decision
         → Blink Detection: tracks Eye Aspect Ratio (EAR) via MediaPipe Face Mesh
         → Head Pose Variation: measures micro-movements in yaw/pitch via solvePnP
         → OR logic: pass if EITHER check indicates a live person
         → Fail-safe: if insufficient frames for analysis → ACCESS DENIED

  Edge Cases Handled:
      - Google API mishears the password → Fuzzy matching + keyword backup
      - Dark room / hoarse voice → Gray area with cosine similarity fallback
      - Photo + recording attack → Cross-modal Unknown class detection
      - No face detected → Retry with timeout
      - Microphone failure → Graceful error handling
      - No internet for STT → Clear error message with instructions
      - Model file missing → Startup validation with helpful errors

  Usage:
      python app/run_system.py

  Requirements:
      pip install SpeechRecognition sounddevice scipy numpy torch torchaudio
      pip install facenet-pytorch opencv-python Pillow speechbrain
      pip install mediapipe  (for liveness detection — Phase 6)

  Note: Google Speech Recognition requires an internet connection.
        For offline alternative, see the Whisper integration notes at bottom.

=============================================================================
"""

import os
import sys
import time
import json
import wave
import tempfile
import warnings
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher

# Suppress non-critical warnings (SpeechBrain is noisy)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
#  PATH SETUP — Ensure imports work from any working directory
# ============================================================
# Add project root to sys.path so we can import utils.config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
#  CONFIGURATION — Import from centralized config
# ============================================================
try:
    from utils.config import (
        PROJECT_ROOT as _PROJECT_ROOT,
        PATHS, CLASSES, THRESHOLDS, LIVE_SYSTEM,
        EMBEDDINGS, TRAINING, DEVICE as DEVICE_CONFIG,
        FACE_PREPROCESS, VOICE_PREPROCESS, CAPTURE,
        LIVENESS
    )
    print("[CONFIG] Loaded configuration from utils/config.py")

    # --- Map class-based config to flat variables used by this script ---

    # Device
    DEVICE = DEVICE_CONFIG.COMPUTE

    # Paths
    FUSION_MODEL_PATH = PATHS.FUSION_MODEL
    USER_PROFILES_PATH = PATHS.USER_PROFILES

    # Class labels — MUST match the order used during training
    CLASS_LABELS = CLASSES.ALL_CLASSES
    NUM_CLASSES = CLASSES.NUM_CLASSES

    # Embedding dimensions
    FACE_EMBEDDING_DIM = EMBEDDINGS.FACE_EMBEDDING_DIM
    VOICE_EMBEDDING_DIM = EMBEDDINGS.VOICE_EMBEDDING_DIM
    FUSED_DIM = EMBEDDINGS.FUSED_EMBEDDING_DIM  # 704

    # Decision thresholds
    CONFIDENCE_HIGH = THRESHOLDS.HIGH_CONFIDENCE       # ≥ this → immediate access
    CONFIDENCE_LOW = THRESHOLDS.LOW_CONFIDENCE          # < this → reject

    # Cosine similarity thresholds for gray area fallback
    FACE_SIM_THRESHOLD = THRESHOLDS.FACE_SIMILARITY_MIN
    VOICE_SIM_THRESHOLD = THRESHOLDS.VOICE_SIMILARITY_MIN

    # Password settings
    VOICE_PASSWORD = LIVE_SYSTEM.PASSWORD_PHRASE
    PASSWORD_FUZZY_THRESHOLD = THRESHOLDS.PASSWORD_FUZZY_THRESHOLD
    PASSWORD_KEYWORDS = THRESHOLDS.PASSWORD_KEYWORDS

    # Audio recording settings
    SAMPLE_RATE = LIVE_SYSTEM.RECORDING_SAMPLE_RATE
    VOICE_RECORD_DURATION = LIVE_SYSTEM.RECORDING_DURATION_SEC
    PASSWORD_RECORD_DURATION = LIVE_SYSTEM.PASSWORD_RECORD_DURATION

    # Face detection settings
    FACE_DETECTION_TIMEOUT = LIVE_SYSTEM.FACE_DETECTION_TIMEOUT
    FACE_DETECTION_CONFIDENCE = LIVE_SYSTEM.LIVE_FACE_CONFIDENCE  # 0.90 for live (relaxed)
    MAX_FACE_RETRIES = LIVE_SYSTEM.MAX_FACE_RETRIES

    # Retry settings
    MAX_PASSWORD_ATTEMPTS = LIVE_SYSTEM.MAX_ATTEMPTS
    LOCKOUT_DURATION_SEC = LIVE_SYSTEM.LOCKOUT_DURATION_SEC

    # Admin mode
    ADMIN_KEY = LIVE_SYSTEM.ADMIN_KEY

    # Logging
    LOG_DIR = LIVE_SYSTEM.LOG_DIR
    LOG_ATTEMPTS = LIVE_SYSTEM.LOG_ATTEMPTS

    # Camera
    CAMERA_INDEX = LIVE_SYSTEM.CAMERA_INDEX
    CAMERA_WARMUP_SEC = LIVE_SYSTEM.CAMERA_WARMUP_SEC

    # Capture & Preview settings
    SHOW_PREVIEW = CAPTURE.SHOW_PREVIEW
    PREVIEW_WINDOW_NAME = CAPTURE.PREVIEW_WINDOW_NAME
    FACE_CAPTURE_DURATION = CAPTURE.FACE_CAPTURE_DURATION
    FACE_TOP_K = CAPTURE.FACE_TOP_K_FRAMES
    FACE_MIN_QUALITY = CAPTURE.FACE_MIN_QUALITY
    QUALITY_W_CONFIDENCE = CAPTURE.QUALITY_WEIGHT_CONFIDENCE
    QUALITY_W_SIZE = CAPTURE.QUALITY_WEIGHT_SIZE
    QUALITY_W_CENTER = CAPTURE.QUALITY_WEIGHT_CENTER
    CAPTURE_BOX_GOOD = CAPTURE.BOX_COLOR_GOOD
    CAPTURE_BOX_UNCERTAIN = CAPTURE.BOX_COLOR_UNCERTAIN
    CAPTURE_BOX_NONE = CAPTURE.BOX_COLOR_NONE
    CAPTURE_BOX_THICKNESS = CAPTURE.BOX_THICKNESS
    CAPTURE_VOICE_BAR_H = CAPTURE.VOICE_BAR_HEIGHT
    CAPTURE_VOICE_BAR_COLOR = CAPTURE.VOICE_BAR_COLOR
    CAPTURE_VOICE_BAR_BG = CAPTURE.VOICE_BAR_BG_COLOR

    # Liveness detection settings
    LIVENESS_ENABLED = LIVENESS.ENABLED
    LIVENESS_EAR_THRESHOLD = LIVENESS.EAR_THRESHOLD
    LIVENESS_BLINK_CONSEC = LIVENESS.BLINK_CONSEC_FRAMES
    LIVENESS_MIN_BLINKS = LIVENESS.MIN_BLINKS_REQUIRED
    LIVENESS_EAR_VAR_THRESHOLD = LIVENESS.EAR_VARIATION_THRESHOLD
    LIVENESS_POSE_MIN_STD_YAW = LIVENESS.HEAD_POSE_MIN_STD_YAW
    LIVENESS_POSE_MIN_STD_PITCH = LIVENESS.HEAD_POSE_MIN_STD_PITCH
    LIVENESS_REQUIRE_ALL = LIVENESS.REQUIRE_ALL
    LIVENESS_MIN_FRAMES = LIVENESS.MIN_FRAMES_FOR_ANALYSIS
    LIVENESS_MESH_MAX_FACES = LIVENESS.FACE_MESH_MAX_FACES
    LIVENESS_MESH_DET_CONF = LIVENESS.FACE_MESH_DETECTION_CONF
    LIVENESS_MESH_PRESENCE_CONF = LIVENESS.FACE_MESH_PRESENCE_CONF
    LIVENESS_MESH_TRACK_CONF = LIVENESS.FACE_MESH_TRACKING_CONF
    LIVENESS_MODEL_FILENAME = LIVENESS.FACE_LANDMARKER_MODEL
    LIVENESS_MODEL_URL = LIVENESS.FACE_LANDMARKER_URL
    LIVENESS_LEFT_EYE = LIVENESS.LEFT_EYE
    LIVENESS_RIGHT_EYE = LIVENESS.RIGHT_EYE
    LIVENESS_POSE_LANDMARKS = LIVENESS.POSE_LANDMARKS

except ImportError:
    print("[CONFIG] WARNING: Could not import utils/config.py — using built-in defaults")

    import torch as _torch_init
    DEVICE = _torch_init.device("cuda:0" if _torch_init.cuda.is_available() else "cpu")

    FUSION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fusion_model.pt")
    USER_PROFILES_PATH = os.path.join(PROJECT_ROOT, "models", "user_profiles.pt")

    CLASS_LABELS = ["david", "itzhak", "yossi", "unknown"]
    NUM_CLASSES = len(CLASS_LABELS)

    FACE_EMBEDDING_DIM = 512
    VOICE_EMBEDDING_DIM = 192
    FUSED_DIM = FACE_EMBEDDING_DIM + VOICE_EMBEDDING_DIM

    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_LOW = 0.50

    FACE_SIM_THRESHOLD = 0.4
    VOICE_SIM_THRESHOLD = 0.4

    VOICE_PASSWORD = "my voice is my password"
    PASSWORD_FUZZY_THRESHOLD = 0.75
    PASSWORD_KEYWORDS = ["voice", "password"]

    SAMPLE_RATE = 16000
    VOICE_RECORD_DURATION = 6
    PASSWORD_RECORD_DURATION = 5

    FACE_DETECTION_TIMEOUT = 15
    FACE_DETECTION_CONFIDENCE = 0.90
    MAX_FACE_RETRIES = 3

    MAX_PASSWORD_ATTEMPTS = 3
    LOCKOUT_DURATION_SEC = 30

    ADMIN_KEY = "a"

    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    LOG_ATTEMPTS = True

    CAMERA_INDEX = 0
    CAMERA_WARMUP_SEC = 1.0

    # Capture & Preview defaults
    SHOW_PREVIEW = True
    PREVIEW_WINDOW_NAME = "Secure Gateway - Live Preview"
    FACE_CAPTURE_DURATION = 3
    FACE_TOP_K = 3
    FACE_MIN_QUALITY = 0.3
    QUALITY_W_CONFIDENCE = 0.4
    QUALITY_W_SIZE = 0.3
    QUALITY_W_CENTER = 0.3
    CAPTURE_BOX_GOOD = (0, 255, 0)
    CAPTURE_BOX_UNCERTAIN = (0, 165, 255)
    CAPTURE_BOX_NONE = (0, 0, 255)
    CAPTURE_BOX_THICKNESS = 2
    CAPTURE_VOICE_BAR_H = 30
    CAPTURE_VOICE_BAR_COLOR = (0, 200, 0)
    CAPTURE_VOICE_BAR_BG = (40, 40, 40)

    # Liveness fallback defaults
    LIVENESS_ENABLED = True
    LIVENESS_EAR_THRESHOLD = 0.22
    LIVENESS_BLINK_CONSEC = 2
    LIVENESS_MIN_BLINKS = 1
    LIVENESS_EAR_VAR_THRESHOLD = 0.003
    LIVENESS_POSE_MIN_STD_YAW = 0.3
    LIVENESS_POSE_MIN_STD_PITCH = 0.3
    LIVENESS_REQUIRE_ALL = False
    LIVENESS_MIN_FRAMES = 10
    LIVENESS_MESH_MAX_FACES = 1
    LIVENESS_MESH_DET_CONF = 0.5
    LIVENESS_MESH_PRESENCE_CONF = 0.5
    LIVENESS_MESH_TRACK_CONF = 0.5
    LIVENESS_MODEL_FILENAME = "face_landmarker.task"
    LIVENESS_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    LIVENESS_LEFT_EYE = [33, 160, 158, 133, 153, 144]
    LIVENESS_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LIVENESS_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


# ============================================================
#  LAZY IMPORTS — Heavy libraries loaded only when needed
# ============================================================
# This keeps startup fast and provides clear error messages if
# a dependency is missing.

_torch = None
_cv2 = None
_sd = None
_sr = None
_mtcnn = None
_facenet = None
_ecapa = None


def _import_torch():
    """Import PyTorch (usually already imported via config, but just in case)."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_cv2():
    """Import OpenCV for webcam capture."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            print("\n[ERROR] OpenCV not installed!")
            print("  Fix: pip install opencv-python")
            sys.exit(1)
    return _cv2


def _import_sounddevice():
    """Import sounddevice for microphone recording."""
    global _sd
    if _sd is None:
        try:
            import sounddevice as sd
            _sd = sd
        except ImportError:
            print("\n[ERROR] sounddevice not installed!")
            print("  Fix: pip install sounddevice")
            sys.exit(1)
    return _sd


def _import_speech_recognition():
    """Import SpeechRecognition for Google STT API."""
    global _sr
    if _sr is None:
        try:
            import speech_recognition as sr
            _sr = sr
        except ImportError:
            print("\n[ERROR] SpeechRecognition not installed!")
            print("  Fix: pip install SpeechRecognition")
            sys.exit(1)
    return _sr


_mp = None

def _import_mediapipe():
    """Import MediaPipe for liveness detection (face mesh landmarks)."""
    global _mp
    if _mp is None:
        try:
            import mediapipe as mp
            _mp = mp
        except ImportError:
            return None  # Graceful — caller handles this
    return _mp


# ============================================================
#  MODEL DEFINITION — Must match train_model.py architecture
# ============================================================
# We redefine the model class here so run_system.py can load
# the trained weights without importing from training/.
# This is intentional: the app should be self-contained and not
# depend on the training code at runtime.

def _build_fusion_model():
    """
    Recreate the Late Fusion MLP architecture.
    MUST match the architecture in training/train_model.py exactly.

    Architecture:
        Input(704) → Linear(704→256) → BatchNorm → ReLU → Dropout(0.3)
                   → Linear(256→128)  → BatchNorm → ReLU → Dropout(0.2)
                   → Linear(128→NUM_CLASSES) → Softmax (applied during inference)
    """
    torch = _import_torch()
    import torch.nn as nn

    class FusionMLP(nn.Module):
        def __init__(self, input_dim=FUSED_DIM, num_classes=NUM_CLASSES):
            super(FusionMLP, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.network(x)

    return FusionMLP


# ============================================================
#  SECTION 1: STARTUP VALIDATION
# ============================================================
# Before the system goes live, verify that all required files
# and hardware are available. Better to fail fast at startup
# than crash during an authentication attempt.

def validate_system():
    """
    Pre-flight check: verify all models, profiles, and hardware exist.
    Returns True if everything is ready, False otherwise.
    """
    torch = _import_torch()
    print("\n" + "=" * 60)
    print("  SECURE GATEWAY — System Startup Validation")
    print("=" * 60)

    all_ok = True

    # 1. Check model file
    print(f"\n  [1/7] Fusion model ......... ", end="")
    if os.path.exists(FUSION_MODEL_PATH):
        size_mb = os.path.getsize(FUSION_MODEL_PATH) / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
    else:
        print(f"MISSING!")
        print(f"        Expected at: {FUSION_MODEL_PATH}")
        print(f"        Run training/train_model.py first.")
        all_ok = False

    # 2. Check enrollment profiles
    print(f"  [2/7] User profiles ........ ", end="")
    if os.path.exists(USER_PROFILES_PATH):
        profiles = torch.load(USER_PROFILES_PATH, map_location="cpu", weights_only=False)
        enrolled_users = [k for k in profiles.keys() if k != "unknown"]
        print(f"OK ({len(enrolled_users)} users: {', '.join(enrolled_users)})")
    else:
        print(f"MISSING!")
        print(f"        Expected at: {USER_PROFILES_PATH}")
        print(f"        Run data_preparation/enroll_users.py first.")
        all_ok = False

    # 3. Check webcam
    print(f"  [3/7] Webcam ............... ", end="")
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            h, w = frame.shape[:2]
            print(f"OK ({w}x{h})")
        else:
            print(f"OPENED but cannot read frames")
            all_ok = False
    else:
        print(f"NOT FOUND")
        print(f"        Make sure a webcam is connected.")
        all_ok = False

    # 4. Check microphone
    print(f"  [4/7] Microphone ........... ", end="")
    sd = _import_sounddevice()
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            default = sd.query_devices(kind='input')
            print(f"OK ({default['name'][:40]})")
        else:
            print(f"NO INPUT DEVICE FOUND")
            all_ok = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_ok = False

    # 5. Check internet (needed for Google STT)
    print(f"  [5/7] Internet (for STT) ... ", end="")
    try:
        import urllib.request
        urllib.request.urlopen("https://www.google.com", timeout=3)
        print(f"OK")
    except Exception:
        print(f"NO CONNECTION")
        print(f"        Google Speech-to-Text requires internet.")
        print(f"        The system will still run but password check may fail.")
        # Don't set all_ok = False — system can work without STT in demo mode

    # 6. Check device (GPU/CPU)
    print(f"  [6/7] Compute device ....... ", end="")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU ({gpu_name})")
    else:
        print(f"CPU (slower but functional)")

    # 7. Check liveness detection (MediaPipe)
    print(f"  [7/7] Liveness detection ... ", end="")
    if not LIVENESS_ENABLED:
        print(f"DISABLED (skipped)")
    else:
        mp = _import_mediapipe()
        if mp is not None:
            # Also check if the model file exists (will be downloaded at load time if not)
            model_dir = PATHS.MODELS_DIR if hasattr(PATHS, 'MODELS_DIR') else os.path.join(PROJECT_ROOT, "models")
            model_path = os.path.join(model_dir, LIVENESS_MODEL_FILENAME)
            model_note = "" if os.path.exists(model_path) else " (model will be downloaded)"
            print(f"OK (MediaPipe {mp.__version__}){model_note}")
        else:
            print(f"MISSING — MediaPipe not installed!")
            print(f"        Liveness is enabled but MediaPipe is not available.")
            print(f"        Fix: pip install mediapipe")
            print(f"        Or set LIVENESS.ENABLED = False in config.py")
            all_ok = False

    print()
    if all_ok:
        print("  ✓ All systems ready. Starting authentication loop...")
    else:
        print("  ✗ Some components are missing. Fix the issues above first.")

    print("=" * 60)
    return all_ok


# ============================================================
#  SECTION 2: MODEL LOADING
# ============================================================

class ModelManager:
    """
    Loads and manages all AI models needed for inference.
    Models are loaded once at startup and reused for every authentication attempt.
    """

    def __init__(self):
        self.torch = _import_torch()
        self.fusion_model = None
        self.user_profiles = None
        self.mtcnn = None
        self.facenet = None
        self.ecapa = None
        self.face_mesh = None
        self._loaded = False

    def load_all(self):
        """Load all models into memory. Call once at startup."""
        print("\n  Loading AI models (this may take a moment on first run)...")
        start = time.time()

        self._load_fusion_model()
        self._load_user_profiles()
        self._load_face_models()
        self._load_voice_model()
        self._load_mediapipe()

        elapsed = time.time() - start
        self._loaded = True
        print(f"  ✓ All models loaded in {elapsed:.1f}s")
        print(f"  ✓ Running on: {DEVICE}\n")

    def _load_fusion_model(self):
        """Load the trained Fusion MLP from fusion_model.pt."""
        print(f"    → Fusion MLP ............ ", end="", flush=True)
        FusionMLP = _build_fusion_model()
        self.fusion_model = FusionMLP(input_dim=FUSED_DIM, num_classes=NUM_CLASSES)

        # Load saved weights
        checkpoint = self.torch.load(FUSION_MODEL_PATH, map_location=DEVICE, weights_only=False)

        # Handle both formats: raw state_dict or wrapped checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.fusion_model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.fusion_model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the file IS the state_dict
            self.fusion_model.load_state_dict(checkpoint)

        self.fusion_model.to(DEVICE)
        self.fusion_model.eval()
        print("OK")

    def _load_user_profiles(self):
        """Load enrollment profiles (mean embeddings per user)."""
        print(f"    → User profiles ......... ", end="", flush=True)
        self.user_profiles = self.torch.load(USER_PROFILES_PATH, map_location=DEVICE, weights_only=False)
        users = [k for k in self.user_profiles.keys() if k != "unknown"]
        print(f"OK ({len(users)} enrolled)")

    def _load_face_models(self):
        """Load MTCNN (face detection) and FaceNet (face embedding)."""
        print(f"    → MTCNN + FaceNet ....... ", end="", flush=True)
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.mtcnn = MTCNN(
            image_size=160,
            margin=40,
            min_face_size=40,
            thresholds=[0.6, 0.7, FACE_DETECTION_CONFIDENCE],
            keep_all=False,
            select_largest=True,
            device=DEVICE
        )

        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        print("OK")

    def _load_voice_model(self):
        """Load ECAPA-TDNN (speaker verification) from SpeechBrain."""
        print(f"    → ECAPA-TDNN ............ ", end="", flush=True)

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        # Use LocalStrategy.COPY to avoid Windows symlink errors (WinError 1314).
        # Windows blocks symlink creation unless running as admin.
        # COPY physically copies the model files instead of symlinking.
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(PROJECT_ROOT, "models", "voice", "ecapa_tdnn"),
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": str(DEVICE)}
        )
        print("OK")

    def _load_mediapipe(self):
        """Load MediaPipe FaceLandmarker for liveness detection (Tasks API)."""
        if not LIVENESS_ENABLED:
            print(f"    → MediaPipe FaceLandmarker  SKIPPED (liveness disabled)")
            return

        print(f"    → MediaPipe FaceLandmarker  ", end="", flush=True)
        mp = _import_mediapipe()

        if mp is None:
            print("NOT AVAILABLE (pip install mediapipe)")
            print("      ⚠ Liveness detection will deny all attempts (fail-safe)")
            return

        # Ensure the model file exists — download if needed
        model_path = os.path.join(PATHS.MODELS_DIR if hasattr(PATHS, 'MODELS_DIR')
                                  else os.path.join(PROJECT_ROOT, "models"),
                                  LIVENESS_MODEL_FILENAME)

        if not os.path.exists(model_path):
            print(f"downloading model...", end=" ", flush=True)
            try:
                import urllib.request
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(LIVENESS_MODEL_URL, model_path)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"({size_mb:.1f} MB) ", end="", flush=True)
            except Exception as e:
                print(f"DOWNLOAD FAILED: {e}")
                print("      ⚠ Download the model manually from:")
                print(f"        {LIVENESS_MODEL_URL}")
                print(f"        Save as: {model_path}")
                return

        # Create FaceLandmarker with IMAGE mode (we process frames one by one)
        # NOTE: We load the model as bytes (model_asset_buffer) instead of by path
        # (model_asset_path) because MediaPipe's C++ backend cannot handle
        # non-ASCII characters in file paths (e.g., Hebrew folder names on Windows).
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        with open(model_path, "rb") as f:
            model_data = f.read()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=LIVENESS_MESH_MAX_FACES,
            min_face_detection_confidence=LIVENESS_MESH_DET_CONF,
            min_face_presence_confidence=LIVENESS_MESH_PRESENCE_CONF,
            min_tracking_confidence=LIVENESS_MESH_TRACK_CONF,
        )

        self.face_mesh = FaceLandmarker.create_from_options(options)
        print("OK")

    def cleanup(self):
        """
        Release all models and free GPU memory.
        Called during graceful shutdown (Ctrl+C).
        """
        print("  Releasing AI models...", end=" ", flush=True)

        # Delete model references
        self.fusion_model = None
        self.user_profiles = None
        self.mtcnn = None
        self.facenet = None
        self.ecapa = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
        self._loaded = False

        # Free GPU memory
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        print("OK")


# ============================================================
#  SECTION 3: AUDIO RECORDING
# ============================================================

def record_audio(duration_sec, sample_rate=SAMPLE_RATE, prompt="Recording"):
    """
    Record audio from the default microphone.

    Args:
        duration_sec: How long to record (seconds).
        sample_rate: Sample rate in Hz (default 16000 for speech models).
        prompt: What to display during recording.

    Returns:
        numpy.ndarray: Audio waveform (mono, float32, normalized).
        None if recording failed.
    """
    sd = _import_sounddevice()

    print(f"\n    🎤 {prompt}... ({duration_sec}s)")
    print(f"       ", end="", flush=True)

    try:
        # Record audio
        audio = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )

        # Show countdown timer
        for i in range(duration_sec):
            time.sleep(1)
            remaining = duration_sec - i - 1
            if remaining > 0:
                print(f"[{remaining}s] ", end="", flush=True)
            else:
                print("[Done]", flush=True)

        sd.wait()  # Block until recording is finished

        # Flatten to 1D
        audio = audio.flatten()

        # Check if we actually captured audio (not silence)
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        if rms_db < -50:
            print(f"    ⚠ WARNING: Very low audio level ({rms_db:.1f} dB)")
            print(f"      Make sure the microphone is working and you're speaking loudly enough.")
            # Don't return None — let the system try with what it has

        return audio

    except Exception as e:
        print(f"\n    ✗ Recording failed: {e}")
        print(f"      Check that your microphone is connected and not in use by another app.")
        return None


def save_audio_to_wav(audio, filepath, sample_rate=SAMPLE_RATE):
    """
    Save a numpy audio array to a WAV file.
    Needed because SpeechRecognition reads from files, not raw arrays.
    """
    import scipy.io.wavfile as wavfile

    # Ensure audio is in int16 format for WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filepath, sample_rate, audio_int16)


# ============================================================
#  SECTION 4: PASSWORD VERIFICATION (Gate 1)
# ============================================================
# Uses Google Speech-to-Text API to transcribe the spoken password,
# then applies fuzzy matching to handle imperfect transcriptions.
#
# Two layers of matching:
#   1. Full phrase fuzzy match (SequenceMatcher ratio ≥ 75%)
#   2. Keyword backup (if key words like "voice" and "password" appear)
#
# This handles cases like:
#   - "my voice is my passport" (close enough → accepted)
#   - "voice ... password" (keywords found → accepted)
#   - "hello world" (nothing matches → rejected)

def verify_password(audio_data):
    """
    Transcribe the spoken audio and verify it matches the stored password.

    Args:
        audio_data: numpy array of recorded audio.

    Returns:
        dict with:
            - passed (bool): Whether the password was accepted.
            - transcript (str): What Google heard.
            - method (str): How it was accepted ("fuzzy_match" / "keyword_match" / "rejected").
            - score (float): Similarity score (0-1).
    """
    sr = _import_speech_recognition()

    result = {
        "passed": False,
        "transcript": "",
        "method": "rejected",
        "score": 0.0
    }

    # Save audio to a temporary WAV file for SpeechRecognition
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        save_audio_to_wav(audio_data, tmp_path)

    try:
        # Load the WAV file into SpeechRecognition
        recognizer = sr.Recognizer()

        with sr.AudioFile(tmp_path) as source:
            # Adjust for ambient noise (brief, since we already normalized)
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.record(source)

        # Transcribe using Google Speech-to-Text API
        print(f"    📡 Sending to Google STT API...", end=" ", flush=True)

        try:
            transcript = recognizer.recognize_google(audio, language="en-US")
        except sr.UnknownValueError:
            print("Could not understand audio")
            print(f"    ⚠ Google could not understand the audio.")
            print(f"      Try speaking more clearly and closer to the microphone.")
            return result
        except sr.RequestError as e:
            print(f"API Error")
            print(f"    ✗ Google STT API error: {e}")
            print(f"      Check your internet connection.")
            return result

        transcript = transcript.strip().lower()
        result["transcript"] = transcript
        print(f'Heard: "{transcript}"')

        # --- Layer 1: Full Phrase Fuzzy Match ---
        # Compare the entire transcription against the stored password
        password_lower = VOICE_PASSWORD.lower().strip()
        similarity = SequenceMatcher(None, transcript, password_lower).ratio()
        result["score"] = similarity

        if similarity >= PASSWORD_FUZZY_THRESHOLD:
            result["passed"] = True
            result["method"] = "fuzzy_match"
            print(f"    ✓ Password ACCEPTED (fuzzy match: {similarity:.0%})")
            return result

        # --- Layer 2: Keyword Backup ---
        # Even if the full phrase didn't match, check if key words are present.
        # This handles cases where Google garbles the middle but gets the important words.
        keywords_found = 0
        for keyword in PASSWORD_KEYWORDS:
            if keyword.lower() in transcript:
                keywords_found += 1

        if keywords_found >= len(PASSWORD_KEYWORDS):
            result["passed"] = True
            result["method"] = "keyword_match"
            result["score"] = max(similarity, 0.75)  # Override score since keywords matched
            print(f"    ✓ Password ACCEPTED (keyword backup: found {keywords_found}/{len(PASSWORD_KEYWORDS)} keywords)")
            return result

        # --- Both failed ---
        print(f"    ✗ Password REJECTED (similarity: {similarity:.0%}, keywords: {keywords_found}/{len(PASSWORD_KEYWORDS)})")
        return result

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ============================================================
#  SECTION 5: CAPTURE SESSION — Camera, Preview & Face Quality
# ============================================================
# Manages the webcam, live preview window, voice amplitude bar,
# multi-frame face capture with quality scoring, and best-frame
# selection. Created fresh for each authentication attempt.
#
# Features:
#   - Live camera preview with bounding boxes (green/orange/red)
#   - Voice amplitude bar during password recording
#   - Multi-frame capture: records 3s of video after first face detected
#   - Quality scoring: confidence × 0.4 + size × 0.3 + centrality × 0.3
#   - Top-K selection: picks the best frame for embedding
#   - Graceful cancel if user closes the preview window
#
# When SHOW_PREVIEW is False, everything works the same but without
# the OpenCV window — purely terminal-based like the original system.

class CaptureSession:
    """
    Manages a single authentication attempt's camera and preview window.

    Lifecycle:
        session = CaptureSession(models)          # opens camera + window
        audio   = session.run_password_phase(5)    # records audio, shows voice bar
        cands   = session.run_face_capture_phase()  # multi-frame capture + scoring
        emb, fr = session.get_best_face_embedding(cands)  # embed top frame
        session.close()                            # release everything

    If the user closes the preview window mid-attempt, session.cancelled
    becomes True and the caller should abort the current attempt.
    """

    def __init__(self, models, show_preview=SHOW_PREVIEW):
        self.models = models
        self.show_preview = show_preview
        self.cap = None
        self.window_open = False
        self._cancelled = False
        self._open_camera()
        if self.show_preview:
            self._open_window()

    # ----------------------------------------------------------
    #  Camera & Window Management
    # ----------------------------------------------------------

    def _open_camera(self):
        """Open the webcam and set resolution."""
        cv2 = _import_cv2()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam (index={CAMERA_INDEX})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(CAMERA_WARMUP_SEC)

    def _open_window(self):
        """Create the OpenCV preview window."""
        cv2 = _import_cv2()
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        self.window_open = True

    def _is_window_open(self):
        """Check if the user has closed the preview window."""
        if not self.show_preview or not self.window_open:
            return True  # No preview → always "open"
        cv2 = _import_cv2()
        try:
            prop = cv2.getWindowProperty(PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            return prop >= 1
        except Exception:
            return False

    @property
    def cancelled(self):
        """True if the user closed the preview window."""
        return self._cancelled

    def close(self):
        """Release camera and destroy preview window."""
        cv2 = _import_cv2()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.show_preview and self.window_open:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                pass
            self.window_open = False

    # ----------------------------------------------------------
    #  Face Detection (boxes only — fast, no cropping)
    # ----------------------------------------------------------

    def _detect_face_box(self, frame_rgb):
        """
        Run MTCNN.detect() to get bounding box + probability.
        This is fast because it only does detection, not alignment/cropping.
        The full MTCNN pipeline runs later only on the best frame.

        Returns:
            box: [x1, y1, x2, y2] numpy array or None
            prob: float detection confidence or None
        """
        from PIL import Image
        pil_image = Image.fromarray(frame_rgb)

        try:
            boxes, probs = self.models.mtcnn.detect(pil_image)
        except Exception:
            return None, None

        if boxes is not None and len(boxes) > 0 and probs is not None:
            # Pick the largest face (most likely the person at the camera)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))
            return boxes[best_idx].copy(), float(probs[best_idx])

        return None, None

    # ----------------------------------------------------------
    #  Quality Scoring
    # ----------------------------------------------------------

    def _compute_quality_score(self, box, prob, frame_shape):
        """
        Score a detected face for multi-frame selection.

        Combines three factors:
          1. MTCNN confidence (0-1): higher = clearer face features
          2. Face size / frame area: larger = closer = more detail
          3. Centrality: closer to frame center = better positioned

        Returns:
            float: quality score (0-1)
        """
        h, w = frame_shape[:2]

        # 1. Detection confidence (already 0-1)
        conf_score = float(prob)

        # 2. Face size relative to frame
        x1, y1, x2, y2 = box
        face_area = max(0, (x2 - x1) * (y2 - y1))
        frame_area = w * h
        size_ratio = face_area / frame_area if frame_area > 0 else 0
        # Normalize: 15% of frame = score 1.0, larger still caps at 1.0
        size_score = min(size_ratio / 0.15, 1.0)

        # 3. How centered the face is
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2
        frame_cx = w / 2
        frame_cy = h / 2
        dist = np.sqrt((face_cx - frame_cx) ** 2 + (face_cy - frame_cy) ** 2)
        max_dist = np.sqrt(frame_cx ** 2 + frame_cy ** 2)
        center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0

        # Weighted combination
        quality = (
            QUALITY_W_CONFIDENCE * conf_score +
            QUALITY_W_SIZE * size_score +
            QUALITY_W_CENTER * center_score
        )

        return quality

    # ----------------------------------------------------------
    #  Preview Drawing
    # ----------------------------------------------------------

    def _draw_frame(self, frame, box=None, prob=None,
                    phase="idle", elapsed=0, duration=0,
                    amplitude=0.0, best_quality=0.0, recording=False):
        """
        Draw overlays on the preview frame and display it.
        Does nothing if show_preview is False.

        Args:
            frame: BGR numpy array from OpenCV
            box: face bounding box [x1,y1,x2,y2] or None
            prob: MTCNN detection probability or None
            phase: "password" / "face_wait" / "face_capture"
            elapsed: seconds since phase started
            duration: total phase duration
            amplitude: current voice amplitude (0-1 range)
            best_quality: highest quality score seen so far
            recording: True if this frame qualifies for recording
        """
        if not self.show_preview:
            return

        cv2 = _import_cv2()
        display = frame.copy()
        h, w = display.shape[:2]

        # --- Top info bar (black background with white text) ---
        bar_top_h = 45
        cv2.rectangle(display, (0, 0), (w, bar_top_h), (0, 0, 0), -1)

        if phase == "password":
            remaining = max(0, duration - elapsed)
            text = f"SAY THE PASSWORD  ({remaining:.0f}s)"
            cv2.putText(display, text, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif phase == "face_wait":
            remaining = max(0, duration - elapsed)
            text = f"WAITING FOR FACE...  ({remaining:.0f}s)"
            cv2.putText(display, text, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        elif phase == "face_capture":
            remaining = max(0, duration - elapsed)
            text = f"CAPTURING  ({remaining:.0f}s)  |  Best: {best_quality:.0%}"
            cv2.putText(display, text, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Face bounding box ---
        if box is not None and prob is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            quality = self._compute_quality_score(box, prob, frame.shape)

            if recording and quality >= FACE_MIN_QUALITY:
                color = CAPTURE_BOX_GOOD       # green
                label = f"Recording  ({prob:.0%})"
            elif prob >= 0.70:
                color = CAPTURE_BOX_UNCERTAIN  # orange
                label = f"Adjust position  ({prob:.0%})"
            else:
                color = CAPTURE_BOX_UNCERTAIN  # orange
                label = f"Low quality  ({prob:.0%})"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, CAPTURE_BOX_THICKNESS)
            # Label background for readability
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 5)
            cv2.rectangle(display, (x1, label_y - label_size[1] - 5),
                          (x1 + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(display, label, (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        elif phase in ("face_wait", "face_capture"):
            # No face — show red warning
            msg = "NO FACE DETECTED"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            tx = (w - text_size[0]) // 2
            ty = (h + text_size[1]) // 2
            cv2.putText(display, msg, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, CAPTURE_BOX_NONE, 2)

        # --- Voice amplitude bar (password phase only) ---
        if phase == "password":
            bar_y = h - CAPTURE_VOICE_BAR_H
            # Background
            cv2.rectangle(display, (0, bar_y), (w, h), CAPTURE_VOICE_BAR_BG, -1)
            # Amplitude fill (scale up for visibility — normal speech is ~0.01-0.1 RMS)
            bar_width = int(w * min(amplitude * 8, 1.0))
            if bar_width > 0:
                cv2.rectangle(display, (0, bar_y), (bar_width, h),
                              CAPTURE_VOICE_BAR_COLOR, -1)
            # Label
            cv2.putText(display, "Voice Level", (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(PREVIEW_WINDOW_NAME, display)
        cv2.waitKey(1)

    # ----------------------------------------------------------
    #  Password Phase — Record audio with camera preview + voice bar
    # ----------------------------------------------------------

    def run_password_phase(self, duration):
        """
        Record password audio while showing camera preview with voice bar.

        Uses sounddevice callback mode for real-time amplitude feedback.
        The camera preview shows the user's face (for reference) and a
        voice amplitude bar at the bottom.

        Args:
            duration: seconds to record

        Returns:
            numpy.ndarray: recorded audio (mono, float32) or None if cancelled/failed
        """
        sd = _import_sounddevice()
        cv2 = _import_cv2()

        print(f"\n    🎤 Say the password now... ({duration}s)")
        print(f"       ", end="", flush=True)

        audio_chunks = []
        current_amplitude = [0.0]

        def audio_callback(indata, frames, time_info, status):
            audio_chunks.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata ** 2)))
            current_amplitude[0] = rms

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=int(SAMPLE_RATE * 0.05)  # 50ms chunks for smooth bar
            )
        except Exception as e:
            print(f"\n    ✗ Microphone error: {e}")
            return None

        start_time = time.time()
        last_print_sec = -1

        with stream:
            while time.time() - start_time < duration:
                # Check if window was closed
                if not self._is_window_open():
                    self._cancelled = True
                    print("\n    ⚠ Preview window closed — attempt cancelled")
                    return None

                # Read camera frame
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Detect face for preview (not storing — just visual feedback)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                box, prob = self._detect_face_box(frame_rgb)

                elapsed = time.time() - start_time

                # Draw preview with voice bar
                self._draw_frame(
                    frame, box, prob,
                    phase="password",
                    elapsed=elapsed,
                    duration=duration,
                    amplitude=current_amplitude[0]
                )

                # Terminal countdown
                sec = int(elapsed)
                if sec > last_print_sec:
                    remaining = duration - sec
                    if remaining > 0:
                        print(f"[{remaining}s] ", end="", flush=True)
                    last_print_sec = sec

        print("[Done]", flush=True)

        # Combine all audio chunks
        if not audio_chunks:
            print(f"    ✗ No audio recorded")
            return None

        audio = np.concatenate(audio_chunks, axis=0).flatten()

        # Check audio level
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        if rms_db < -50:
            print(f"    ⚠ WARNING: Very low audio level ({rms_db:.1f} dB)")
            print(f"      Make sure the microphone is working and you're speaking loudly enough.")

        return audio

    # ----------------------------------------------------------
    #  Face Capture Phase — Multi-frame collection + quality scoring
    # ----------------------------------------------------------

    def run_face_capture_phase(self):
        """
        Two-phase face capture with quality scoring.

        Phase 1 (up to FACE_DETECTION_TIMEOUT seconds):
            Wait for the first face to be detected. Preview shows
            "Waiting for face..." with red text if nothing is found.

        Phase 2 (FACE_CAPTURE_DURATION seconds):
            Once a face is detected, keep recording frames and scoring
            each one. Preview shows green/orange boxes. All frames above
            FACE_MIN_QUALITY are stored as candidates.

        Returns:
            list of candidate dicts sorted by quality (best first).
            Each dict has: frame_rgb, box, prob, quality
            Empty list if no face was detected.
        """
        cv2 = _import_cv2()

        candidates = []
        best_quality = 0.0

        print(f"\n    📸 Starting face capture...")
        print(f"       Look at the camera. Waiting for face detection...")

        # ---- PHASE 1: Wait for first face ----
        phase1_start = time.time()
        first_face_found = False
        search_prints = 0

        while time.time() - phase1_start < FACE_DETECTION_TIMEOUT:
            if not self._is_window_open():
                self._cancelled = True
                print("\n    ⚠ Preview window closed — attempt cancelled")
                return []

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box, prob = self._detect_face_box(frame_rgb)

            elapsed = time.time() - phase1_start

            if box is not None and prob is not None:
                first_face_found = True
                quality = self._compute_quality_score(box, prob, frame.shape)
                candidates.append({
                    'frame_rgb': frame_rgb.copy(),
                    'box': box.copy(),
                    'prob': float(prob),
                    'quality': quality
                })
                best_quality = quality

                self._draw_frame(
                    frame, box, prob,
                    phase="face_capture", elapsed=0,
                    duration=FACE_CAPTURE_DURATION,
                    best_quality=best_quality, recording=True
                )

                print(f"    ✓ Face detected! Collecting best frames "
                      f"for {FACE_CAPTURE_DURATION}s...")
                break

            # Preview: waiting state
            self._draw_frame(
                frame, box, prob,
                phase="face_wait",
                elapsed=elapsed,
                duration=FACE_DETECTION_TIMEOUT
            )

            # Terminal progress (every ~2 seconds)
            secs = int(elapsed)
            if secs >= 2 and secs % 2 == 0 and secs > search_prints:
                remaining = FACE_DETECTION_TIMEOUT - elapsed
                print(f"       ... searching for face ({remaining:.0f}s remaining)")
                search_prints = secs

        if not first_face_found:
            print(f"    ✗ No face detected within {FACE_DETECTION_TIMEOUT}s timeout")
            print(f"      Tips: ensure good lighting, face the camera, remove obstructions")
            return []

        # ---- PHASE 2: Collect frames for quality selection ----
        phase2_start = time.time()
        total_frames = 0

        while time.time() - phase2_start < FACE_CAPTURE_DURATION:
            if not self._is_window_open():
                self._cancelled = True
                print("\n    ⚠ Preview window closed — attempt cancelled")
                break

            ret, frame = self.cap.read()
            if not ret:
                continue

            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box, prob = self._detect_face_box(frame_rgb)

            elapsed = time.time() - phase2_start

            if box is not None and prob is not None:
                quality = self._compute_quality_score(box, prob, frame.shape)

                if quality >= FACE_MIN_QUALITY:
                    candidates.append({
                        'frame_rgb': frame_rgb.copy(),
                        'box': box.copy(),
                        'prob': float(prob),
                        'quality': quality
                    })
                    best_quality = max(best_quality, quality)

                is_recording = quality >= FACE_MIN_QUALITY
                self._draw_frame(
                    frame, box, prob,
                    phase="face_capture",
                    elapsed=elapsed,
                    duration=FACE_CAPTURE_DURATION,
                    best_quality=best_quality,
                    recording=is_recording
                )
            else:
                self._draw_frame(
                    frame, None, None,
                    phase="face_capture",
                    elapsed=elapsed,
                    duration=FACE_CAPTURE_DURATION,
                    best_quality=best_quality
                )

        # Sort by quality (best first)
        candidates.sort(key=lambda c: c['quality'], reverse=True)

        print(f"    ✓ Collected {len(candidates)} candidate frames, "
              f"best quality: {best_quality:.0%} "
              f"(from {total_frames} total)")

        return candidates

    # ----------------------------------------------------------
    #  Best Frame Selection & Embedding
    # ----------------------------------------------------------

    def get_best_face_embedding(self, candidates):
        """
        Run MTCNN + FaceNet on the best candidate frame(s) to get embedding.

        Tries the top-K frames in quality order. If MTCNN fails to align
        the best frame (rare — can happen with edge cases), it falls back
        to the next best.

        Args:
            candidates: list of candidate dicts from run_face_capture_phase()

        Returns:
            face_embedding: torch.Tensor [512] or None
            raw_frame: numpy RGB array of the frame used, or None
        """
        torch = _import_torch()
        from PIL import Image

        if not candidates:
            return None, None

        top_k = candidates[:FACE_TOP_K]

        for i, candidate in enumerate(top_k):
            pil_image = Image.fromarray(candidate['frame_rgb'])

            # Run full MTCNN pipeline (detect → align → crop → 160x160)
            face_tensor = self.models.mtcnn(pil_image)

            if face_tensor is not None:
                face_batch = face_tensor.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    embedding = self.models.facenet(face_batch).squeeze(0)

                quality = candidate['quality']
                prob = candidate['prob']
                print(f"    ✓ Face embedded (512-dim) — "
                      f"quality: {quality:.0%}, confidence: {prob:.0%}"
                      + (f" [used frame #{i+1}]" if i > 0 else ""))
                return embedding, candidate['frame_rgb']

        print(f"    ✗ MTCNN alignment failed on top-{len(top_k)} frames")
        return None, None


# ============================================================
#  SECTION 5B: LIVENESS DETECTION (Anti-Spoofing)
# ============================================================
# Analyzes the captured video frames for signs of a live person
# vs. a static photo or printout.
#
# Runs AFTER face capture completes and BEFORE the fusion model.
# Uses MediaPipe Face Mesh (468 landmarks) for two checks:
#
#   A. BLINK DETECTION — Eye Aspect Ratio (EAR) tracking:
#      - Compute EAR per frame from 6 eye landmarks
#      - Detect blink events (EAR dip below threshold → recovery)
#      - Also check EAR micro-variation (real eyes fluctuate subtly)
#
#   B. HEAD POSE VARIATION — Yaw/Pitch micro-movements:
#      - Estimate head pose per frame via solvePnP
#      - Compute standard deviation of angles across all frames
#      - Real person: involuntary micro-movements (std > 0.3°)
#      - Photo: near-zero variation (only camera sensor noise)
#
# Decision: OR logic — pass if EITHER check indicates life.
# Fail-safe: if too few frames have landmarks → DENY access.

def _compute_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    where:
        p1 = outer corner, p2 = upper outer lid, p3 = upper inner lid,
        p4 = inner corner, p5 = lower inner lid, p6 = lower outer lid

    When open: EAR ≈ 0.25-0.35
    When closed (blink): EAR < 0.20

    Args:
        landmarks: MediaPipe NormalizedLandmarkList
        eye_indices: list of 6 landmark indices [outer, upper2, upper1, inner, lower1, lower2]
        frame_w: frame width in pixels (for denormalization)
        frame_h: frame height in pixels (for denormalization)

    Returns:
        float: EAR value
    """
    # Extract the 6 landmark points as pixel coordinates
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))

    p1, p2, p3, p4, p5, p6 = points

    # Vertical distances (upper-lower pairs)
    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    vertical_1 = dist(p2, p6)  # upper outer to lower outer
    vertical_2 = dist(p3, p5)  # upper inner to lower inner
    horizontal = dist(p1, p4)  # outer corner to inner corner

    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def _estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Estimate head pose (yaw, pitch, roll) using solvePnP.

    Uses 6 facial landmarks to establish 2D-3D correspondence,
    then solves for the rotation that maps a generic 3D face model
    to the observed 2D positions.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList
        frame_w: frame width in pixels
        frame_h: frame height in pixels

    Returns:
        (yaw, pitch, roll) in degrees, or None if estimation fails
    """
    # 3D model points — generic face model (arbitrary scale, centered at nose)
    # These approximate the relative positions of key facial features
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),        # Chin (landmark 152)
        (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 33)
        (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 263)
        (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
        (150.0, -150.0, -125.0),     # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    # 2D image points from MediaPipe landmarks
    image_points = np.array([
        (landmarks[idx].x * frame_w, landmarks[idx].y * frame_h)
        for idx in LIVENESS_POSE_LANDMARKS
    ], dtype=np.float64)

    # Camera intrinsic matrix (approximate — assumes no distortion)
    focal_length = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve for pose
    cv2 = _import_cv2()
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix, then extract Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose rotation matrix to Euler angles
    # Using the projection matrix approach for stable extraction
    proj_matrix = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        np.vstack((proj_matrix, [0, 0, 0, 1]))[:3, :]
    )

    pitch = euler_angles[0, 0]  # Up-down nod
    yaw = euler_angles[1, 0]    # Left-right turn
    roll = euler_angles[2, 0]   # Head tilt

    return (yaw, pitch, roll)


def check_liveness(candidates, models):
    """
    Analyze captured frames for liveness indicators.

    Runs MediaPipe Face Mesh on each captured frame to extract:
      - Per-frame Eye Aspect Ratio (EAR) → blink detection
      - Per-frame head pose (yaw, pitch) → micro-movement detection

    Decision logic (LIVENESS_REQUIRE_ALL = False):
      - Pass if blink check passes OR head pose check passes
      - Blink check passes if: detected ≥ 1 blink, OR EAR std > threshold
      - Pose check passes if: yaw std > threshold OR pitch std > threshold

    Fail-safe: if fewer than LIVENESS_MIN_FRAMES frames have valid
    landmarks, access is DENIED (cannot verify liveness reliably).

    Args:
        candidates: list of candidate dicts from run_face_capture_phase().
                    Each has 'frame_rgb' (numpy array).
        models: ModelManager with loaded face_mesh.

    Returns:
        dict with:
            - passed (bool): Whether liveness check passed
            - blink_count (int): Number of blinks detected
            - ear_std (float): Standard deviation of EAR across frames
            - ear_mean (float): Mean EAR across frames
            - pose_std_yaw (float): Std dev of yaw angle
            - pose_std_pitch (float): Std dev of pitch angle
            - frames_analyzed (int): Number of frames with valid landmarks
            - method (str): Which check(s) passed or why it failed
            - blink_passed (bool): Whether blink check specifically passed
            - pose_passed (bool): Whether pose check specifically passed
    """
    result = {
        "passed": False,
        "blink_count": 0,
        "ear_std": 0.0,
        "ear_mean": 0.0,
        "pose_std_yaw": 0.0,
        "pose_std_pitch": 0.0,
        "frames_analyzed": 0,
        "method": "not_checked",
        "blink_passed": False,
        "pose_passed": False,
    }

    # --- Guard: liveness disabled ---
    if not LIVENESS_ENABLED:
        result["passed"] = True
        result["method"] = "liveness_disabled"
        return result

    # --- Guard: MediaPipe not available ---
    if models.face_mesh is None:
        print(f"    ⚠ MediaPipe not loaded — liveness check cannot proceed")
        result["method"] = "mediapipe_unavailable"
        return result  # passed = False → fail-safe deny

    # --- Guard: no candidates ---
    if not candidates or len(candidates) == 0:
        result["method"] = "no_frames"
        return result  # passed = False → fail-safe deny

    print(f"\n    🔍 Liveness check — analyzing {len(candidates)} frames...")

    # Import mediapipe for mp.Image conversion
    mp = _import_mediapipe()

    # ---- Step 1: Extract landmarks from all candidate frames ----
    ear_values = []          # EAR per frame (average of both eyes)
    yaw_values = []          # Head yaw per frame
    pitch_values = []        # Head pitch per frame

    for candidate in candidates:
        frame_rgb = candidate['frame_rgb']
        h, w = frame_rgb.shape[:2]

        # Convert numpy array to MediaPipe Image (required by Tasks API)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run FaceLandmarker
        try:
            mesh_result = models.face_mesh.detect(mp_image)
        except Exception:
            continue

        if not mesh_result.face_landmarks or len(mesh_result.face_landmarks) == 0:
            continue

        # face_landmarks[0] is a list of NormalizedLandmark objects
        face_landmarks = mesh_result.face_landmarks[0]

        # --- Compute EAR (average of both eyes) ---
        try:
            left_ear = _compute_ear(face_landmarks, LIVENESS_LEFT_EYE, w, h)
            right_ear = _compute_ear(face_landmarks, LIVENESS_RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)
        except (IndexError, ZeroDivisionError):
            pass

        # --- Compute head pose ---
        try:
            pose = _estimate_head_pose(face_landmarks, w, h)
            if pose is not None:
                yaw, pitch, _ = pose
                yaw_values.append(yaw)
                pitch_values.append(pitch)
        except Exception:
            pass

    frames_analyzed = len(ear_values)
    result["frames_analyzed"] = frames_analyzed

    # ---- Step 2: Check minimum frame requirement (FAIL-SAFE) ----
    if frames_analyzed < LIVENESS_MIN_FRAMES:
        result["method"] = f"insufficient_frames ({frames_analyzed}/{LIVENESS_MIN_FRAMES})"
        print(f"    ✗ Liveness FAILED — only {frames_analyzed} usable frames "
              f"(need {LIVENESS_MIN_FRAMES})")
        print(f"      The system cannot verify you are a live person.")
        return result  # passed = False → DENY

    # ---- Step 3: Blink detection ----
    ear_array = np.array(ear_values)
    ear_std = float(np.std(ear_array))
    ear_mean = float(np.mean(ear_array))
    result["ear_std"] = ear_std
    result["ear_mean"] = ear_mean

    # Count blinks: sequences of BLINK_CONSEC consecutive frames with EAR < threshold
    blink_count = 0
    below_count = 0  # consecutive frames with EAR below threshold

    for ear_val in ear_values:
        if ear_val < LIVENESS_EAR_THRESHOLD:
            below_count += 1
        else:
            if below_count >= LIVENESS_BLINK_CONSEC:
                blink_count += 1
            below_count = 0

    # Check if ended mid-blink (eye still closed at last frame)
    if below_count >= LIVENESS_BLINK_CONSEC:
        blink_count += 1

    result["blink_count"] = blink_count

    # Blink check passes if we detected enough blinks OR if EAR has enough variation
    # The variation check catches subtle eye movement even without a full blink
    blink_passed = (blink_count >= LIVENESS_MIN_BLINKS) or (ear_std >= LIVENESS_EAR_VAR_THRESHOLD)
    result["blink_passed"] = blink_passed

    blink_reason = []
    if blink_count >= LIVENESS_MIN_BLINKS:
        blink_reason.append(f"{blink_count} blink(s) detected")
    if ear_std >= LIVENESS_EAR_VAR_THRESHOLD:
        blink_reason.append(f"EAR variation={ear_std:.4f}")

    # ---- Step 4: Head pose variation ----
    pose_passed = False
    pose_reason = []

    if len(yaw_values) >= LIVENESS_MIN_FRAMES:
        yaw_std = float(np.std(yaw_values))
        pitch_std = float(np.std(pitch_values))
        result["pose_std_yaw"] = yaw_std
        result["pose_std_pitch"] = pitch_std

        yaw_ok = yaw_std >= LIVENESS_POSE_MIN_STD_YAW
        pitch_ok = pitch_std >= LIVENESS_POSE_MIN_STD_PITCH

        pose_passed = yaw_ok or pitch_ok
        result["pose_passed"] = pose_passed

        if yaw_ok:
            pose_reason.append(f"yaw_std={yaw_std:.2f}°")
        if pitch_ok:
            pose_reason.append(f"pitch_std={pitch_std:.2f}°")
    else:
        # Not enough pose data — pose check inconclusive
        pose_reason.append(f"insufficient pose frames ({len(yaw_values)})")

    # ---- Step 5: Final decision ----
    if LIVENESS_REQUIRE_ALL:
        # AND logic: both must pass
        result["passed"] = blink_passed and pose_passed
    else:
        # OR logic: either passes
        result["passed"] = blink_passed or pose_passed

    # Build descriptive method string
    methods_passed = []
    if blink_passed:
        methods_passed.append("blink(" + ", ".join(blink_reason) + ")")
    if pose_passed:
        methods_passed.append("pose(" + ", ".join(pose_reason) + ")")

    if result["passed"]:
        result["method"] = " + ".join(methods_passed) if methods_passed else "passed"
    else:
        # Explain why it failed
        fail_reasons = []
        if not blink_passed:
            fail_reasons.append(f"no blinks (count={blink_count}, EAR_std={ear_std:.4f})")
        if not pose_passed:
            fail_reasons.append(
                f"no movement (yaw_std={result['pose_std_yaw']:.2f}°, "
                f"pitch_std={result['pose_std_pitch']:.2f}°)"
            )
        result["method"] = "failed: " + " & ".join(fail_reasons)

    # ---- Step 6: Print results ----
    print(f"    📊 Liveness Analysis ({frames_analyzed} frames):")
    print(f"       Blink:  {blink_count} blink(s) detected  |  "
          f"EAR mean={ear_mean:.3f}, std={ear_std:.4f}  "
          f"{'✓' if blink_passed else '✗'}")
    print(f"       Pose:   yaw_std={result['pose_std_yaw']:.2f}°, "
          f"pitch_std={result['pose_std_pitch']:.2f}°  "
          f"{'✓' if pose_passed else '✗'}")

    logic_label = "AND" if LIVENESS_REQUIRE_ALL else "OR"

    if result["passed"]:
        print(f"    ✓ Liveness PASSED ({logic_label} logic) — {result['method']}")
    else:
        print(f"    ✗ Liveness FAILED ({logic_label} logic) — {result['method']}")
        print(f"      This may indicate a photo or non-live presentation.")

    return result


# ============================================================
#  SECTION 6: VOICE CAPTURE & EMBEDDING
# ============================================================

def capture_voice_embedding(models, audio_data=None):
    """
    Record a voice clip and compute the speaker embedding using ECAPA-TDNN.

    Can either record fresh audio or use pre-recorded audio
    (e.g., reuse the password recording for the voice embedding
    to avoid making the user speak twice).

    Args:
        models: ModelManager instance with loaded ECAPA-TDNN.
        audio_data: Optional pre-recorded audio (numpy array).
                    If None, records fresh audio from the microphone.

    Returns:
        torch.Tensor: Voice embedding (192-dim) or None if failed.
        numpy.ndarray: The raw audio waveform or None.
    """
    torch = _import_torch()
    import torchaudio

    # Either use provided audio or record new
    if audio_data is None:
        audio_data = record_audio(
            VOICE_RECORD_DURATION,
            sample_rate=SAMPLE_RATE,
            prompt="Recording voice for biometric verification"
        )
        if audio_data is None:
            return None, None

    try:
        # Convert numpy → torch tensor
        # ECAPA-TDNN expects [1, num_samples] at 16kHz
        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

        # Resample if needed (should already be 16kHz from our recording settings)
        if SAMPLE_RATE != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=SAMPLE_RATE,
                new_freq=16000
            )
            waveform = resampler(waveform)

        # Compute embedding
        with torch.no_grad():
            voice_embedding = models.ecapa.encode_batch(waveform.to(DEVICE))
            voice_embedding = voice_embedding.squeeze()  # [192]

        print(f"    ✓ Voice embedded ({voice_embedding.shape[0]}-dim)")
        return voice_embedding, audio_data

    except Exception as e:
        print(f"    ✗ Voice embedding failed: {e}")
        return None, None


# ============================================================
#  SECTION 7: FUSION MODEL INFERENCE
# ============================================================

def run_fusion_model(models, face_embedding, voice_embedding):
    """
    Concatenate face + voice embeddings and run through the Fusion MLP.

    Args:
        models: ModelManager instance with loaded Fusion MLP.
        face_embedding: Tensor of shape [512].
        voice_embedding: Tensor of shape [192].

    Returns:
        dict with:
            - predicted_class (str): The predicted user name or "unknown".
            - confidence (float): Softmax probability of the predicted class.
            - all_probabilities (dict): {class_name: probability} for all classes.
            - predicted_index (int): Index of the predicted class.
    """
    torch = _import_torch()

    # Concatenate: [512] + [192] → [704]
    fused = torch.cat([
        face_embedding.to(DEVICE),
        voice_embedding.to(DEVICE)
    ], dim=0)

    # Add batch dimension: [704] → [1, 704]
    fused_batch = fused.unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        logits = models.fusion_model(fused_batch)  # [1, NUM_CLASSES]
        probabilities = torch.softmax(logits, dim=1).squeeze(0)  # [NUM_CLASSES]

    # Get prediction
    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = CLASS_LABELS[predicted_idx.item()]

    # Build probability dict
    all_probs = {}
    for i, label in enumerate(CLASS_LABELS):
        all_probs[label] = probabilities[i].item()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence.item(),
        "all_probabilities": all_probs,
        "predicted_index": predicted_idx.item()
    }


# ============================================================
#  SECTION 8: GRAY AREA — COSINE SIMILARITY FALLBACK
# ============================================================
# When the Fusion Model gives moderate confidence (50-85%),
# we perform a secondary check using the enrollment profiles.
# This is the safety net described in Phase 1.5 of the roadmap.

def cosine_similarity_check(models, face_embedding, voice_embedding, claimed_user):
    """
    Compare live embeddings against the enrollment profile of the claimed user.

    This is the "Gray Area" handler:
    - The model thinks it might be the user, but isn't sure enough.
    - We check: is the live face close to the user's average face?
    - And: is the live voice close to the user's average voice?
    - If BOTH are above threshold → grant access despite model hesitation.

    Args:
        models: ModelManager instance with loaded user profiles.
        face_embedding: Live face embedding [512].
        voice_embedding: Live voice embedding [192].
        claimed_user: The user the model thinks it might be.

    Returns:
        dict with:
            - passed (bool): Whether the similarity check passed.
            - face_sim (float): Cosine similarity for face.
            - voice_sim (float): Cosine similarity for voice.
    """
    torch = _import_torch()
    import torch.nn.functional as F

    result = {
        "passed": False,
        "face_sim": 0.0,
        "voice_sim": 0.0
    }

    # Check if the claimed user exists in profiles
    if claimed_user not in models.user_profiles or claimed_user == "unknown":
        print(f"    ⚠ No enrollment profile for '{claimed_user}' — cannot perform fallback")
        return result

    profile = models.user_profiles[claimed_user]

    # Get the stored mean embeddings
    # enroll_users.py saves profiles with keys: face_mean, voice_mean, fused_mean, etc.
    if isinstance(profile, dict):
        ref_face = profile.get("face_mean", profile.get("face", profile.get("face_embedding", None)))
        ref_voice = profile.get("voice_mean", profile.get("voice", profile.get("voice_embedding", None)))
    else:
        # If the profile is a different format, skip
        print(f"    ⚠ Unrecognized profile format for '{claimed_user}'")
        return result

    if ref_face is None or ref_voice is None:
        print(f"    ⚠ Incomplete profile for '{claimed_user}'")
        return result

    # Move to same device
    ref_face = ref_face.to(DEVICE)
    ref_voice = ref_voice.to(DEVICE)
    live_face = face_embedding.to(DEVICE)
    live_voice = voice_embedding.to(DEVICE)

    # Compute cosine similarity
    # cosine_similarity expects 2D tensors: [1, dim]
    face_sim = F.cosine_similarity(
        live_face.unsqueeze(0), ref_face.unsqueeze(0)
    ).item()

    voice_sim = F.cosine_similarity(
        live_voice.unsqueeze(0), ref_voice.unsqueeze(0)
    ).item()

    result["face_sim"] = face_sim
    result["voice_sim"] = voice_sim

    print(f"    📊 Cosine Similarity Fallback:")
    print(f"       Face  similarity: {face_sim:.3f}  (threshold: {FACE_SIM_THRESHOLD})")
    print(f"       Voice similarity: {voice_sim:.3f}  (threshold: {VOICE_SIM_THRESHOLD})")

    # Both must pass
    if face_sim >= FACE_SIM_THRESHOLD and voice_sim >= VOICE_SIM_THRESHOLD:
        result["passed"] = True
        print(f"    ✓ Fallback PASSED — both modalities above threshold")
    else:
        print(f"    ✗ Fallback FAILED —", end=" ")
        if face_sim < FACE_SIM_THRESHOLD:
            print(f"face too low ({face_sim:.3f} < {FACE_SIM_THRESHOLD})", end=" ")
        if voice_sim < VOICE_SIM_THRESHOLD:
            print(f"voice too low ({voice_sim:.3f} < {VOICE_SIM_THRESHOLD})", end=" ")
        print()

    return result


# ============================================================
#  SECTION 9: ACCESS DECISION ENGINE
# ============================================================
# Implements the 3-tier confidence logic from the roadmap.

def make_access_decision(models, face_embedding, voice_embedding):
    """
    Run the full decision pipeline:
      1. Feed embeddings to fusion model → get prediction + confidence
      2. Apply 3-tier confidence logic
      3. If gray area → cosine similarity fallback

    Args:
        models: ModelManager with all loaded models.
        face_embedding: Live face embedding [512].
        voice_embedding: Live voice embedding [192].

    Returns:
        dict with:
            - access_granted (bool): Final decision.
            - predicted_user (str): Who the system thinks the person is.
            - confidence (float): Model confidence.
            - decision_path (str): How the decision was made.
            - details (dict): Full breakdown.
    """
    print(f"\n  ────── FUSION MODEL INFERENCE ──────")

    # Step 1: Run fusion model
    prediction = run_fusion_model(models, face_embedding, voice_embedding)

    user = prediction["predicted_class"]
    conf = prediction["confidence"]

    print(f"\n    🧠 Model Prediction: {user.upper()}")
    print(f"       Confidence: {conf:.1%}")
    print(f"       All probabilities:")
    for label, prob in sorted(prediction["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        marker = " ◄" if label == user else ""
        print(f"         {label:<10} {prob:6.1%}  {bar}{marker}")

    # Step 2: Apply 3-tier confidence logic
    decision = {
        "access_granted": False,
        "predicted_user": user,
        "confidence": conf,
        "decision_path": "",
        "details": prediction
    }

    # --- Tier 1: HIGH confidence (≥ 85%) → Immediate access ---
    if user != "unknown" and conf >= CONFIDENCE_HIGH:
        decision["access_granted"] = True
        decision["decision_path"] = f"high_confidence ({conf:.1%} ≥ {CONFIDENCE_HIGH:.0%})"
        print(f"\n    🟢 HIGH CONFIDENCE → ACCESS GRANTED")
        return decision

    # --- Tier 3: LOW confidence (< 50%) → Reject ---
    if conf < CONFIDENCE_LOW or user == "unknown":
        decision["access_granted"] = False
        if user == "unknown":
            decision["decision_path"] = f"unknown_class (conf={conf:.1%})"
            print(f"\n    🔴 UNKNOWN CLASS DETECTED → ACCESS DENIED")
            print(f"       The system detected a face+voice mismatch or unrecognized person.")
        else:
            decision["decision_path"] = f"low_confidence ({conf:.1%} < {CONFIDENCE_LOW:.0%})"
            print(f"\n    🔴 LOW CONFIDENCE → ACCESS DENIED")
        return decision

    # --- Tier 2: GRAY AREA (50-85%) → Cosine similarity fallback ---
    print(f"\n    🟡 GRAY AREA ({conf:.1%}) — Activating cosine similarity fallback...")
    fallback = cosine_similarity_check(models, face_embedding, voice_embedding, user)
    decision["details"]["cosine_fallback"] = fallback

    if fallback["passed"]:
        decision["access_granted"] = True
        decision["decision_path"] = (
            f"gray_area_fallback (model={conf:.1%}, "
            f"face_sim={fallback['face_sim']:.3f}, "
            f"voice_sim={fallback['voice_sim']:.3f})"
        )
        print(f"\n    🟢 GRAY AREA + FALLBACK PASSED → ACCESS GRANTED")
    else:
        decision["access_granted"] = False
        decision["decision_path"] = (
            f"gray_area_rejected (model={conf:.1%}, "
            f"face_sim={fallback['face_sim']:.3f}, "
            f"voice_sim={fallback['voice_sim']:.3f})"
        )
        print(f"\n    🔴 GRAY AREA + FALLBACK FAILED → ACCESS DENIED")

    return decision


# ============================================================
#  SECTION 10: ATTEMPT LOGGING
# ============================================================
# Every authentication attempt is logged for security auditing
# and for the Active Learning system (Phase 4 / smart_finetune.py).

def log_attempt(decision, password_result, face_embedding=None, voice_embedding=None):
    """
    Save a log entry for this authentication attempt.
    Used for:
      - Security auditing (who tried to get in, when, result)
      - Active Learning (smart_finetune.py reads these to correct errors)
    """
    if not LOG_ATTEMPTS:
        return

    # Create log directory if needed
    os.makedirs(LOG_DIR, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "access_granted": decision["access_granted"],
        "predicted_user": decision["predicted_user"],
        "confidence": decision["confidence"],
        "decision_path": decision["decision_path"],
        "password_transcript": password_result.get("transcript", ""),
        "password_score": password_result.get("score", 0.0),
        "password_method": password_result.get("method", ""),
    }

    # Add cosine similarity if available
    if "cosine_fallback" in decision.get("details", {}):
        fb = decision["details"]["cosine_fallback"]
        log_entry["face_similarity"] = fb.get("face_sim", 0.0)
        log_entry["voice_similarity"] = fb.get("voice_sim", 0.0)

    # Add liveness data if available
    if "liveness" in decision.get("details", {}):
        lv = decision["details"]["liveness"]
        log_entry["liveness_passed"] = lv.get("passed", False)
        log_entry["liveness_blink_count"] = lv.get("blink_count", 0)
        log_entry["liveness_ear_std"] = lv.get("ear_std", 0.0)
        log_entry["liveness_ear_mean"] = lv.get("ear_mean", 0.0)
        log_entry["liveness_pose_std_yaw"] = lv.get("pose_std_yaw", 0.0)
        log_entry["liveness_pose_std_pitch"] = lv.get("pose_std_pitch", 0.0)
        log_entry["liveness_frames_analyzed"] = lv.get("frames_analyzed", 0)
        log_entry["liveness_method"] = lv.get("method", "")

    # Save to daily log file
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"attempts_{date_str}.jsonl")

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Also save embeddings for Active Learning (if attempt was rejected)
    # smart_finetune.py can use these to correct false rejections
    if not decision["access_granted"] and face_embedding is not None and voice_embedding is not None:
        torch = _import_torch()
        rejected_dir = os.path.join(LOG_DIR, "rejected_embeddings")
        os.makedirs(rejected_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            "face_embedding": face_embedding.cpu(),
            "voice_embedding": voice_embedding.cpu(),
            "predicted_user": decision["predicted_user"],
            "confidence": decision["confidence"],
            "timestamp": timestamp
        }, os.path.join(rejected_dir, f"rejected_{timestamp}.pt"))


# ============================================================
#  SECTION 11: ADMIN MODE (Hook for Phase 4)
# ============================================================

def handle_admin_override(decision, face_embedding, voice_embedding):
    """
    After a rejection, offer the admin the option to correct the decision.
    This saves the rejected embeddings with the correct label for retraining.

    The actual retraining is handled by smart_finetune.py (Phase 4).
    This function just saves the correction data.

    Args:
        decision: The access decision dict.
        face_embedding: The rejected face embedding.
        voice_embedding: The rejected voice embedding.

    Returns:
        bool: True if admin override was triggered.
    """
    torch = _import_torch()

    print(f"\n  ────── ADMIN MODE ──────")
    print(f"  Press '{ADMIN_KEY}' within 5 seconds to override this rejection.")
    print(f"  Press any other key or wait to skip.")

    # Simple timeout-based key detection
    # For Windows compatibility, we use msvcrt; for Unix, we use select
    admin_pressed = False

    try:
        import platform
        if platform.system() == "Windows":
            import msvcrt
            start = time.time()
            while time.time() - start < 5:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if key == ADMIN_KEY:
                        admin_pressed = True
                    break
                time.sleep(0.1)
        else:
            # Unix/Mac
            import select
            start = time.time()
            while time.time() - start < 5:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    key = sys.stdin.readline().strip().lower()
                    if key == ADMIN_KEY:
                        admin_pressed = True
                    break
    except Exception:
        # If key detection fails, skip admin mode
        pass

    if not admin_pressed:
        print(f"  No admin input. Skipping.")
        return False

    # Admin pressed the key — ask which user this actually was
    print(f"\n  ADMIN OVERRIDE ACTIVATED")
    known_users = [label for label in CLASS_LABELS if label != "unknown"]
    print(f"  Who is this person? Enter the number:")
    for i, name in enumerate(known_users):
        print(f"    [{i + 1}] {name}")
    print(f"    [0] Cancel")

    try:
        choice = input(f"  > ").strip()
        choice_num = int(choice)

        if choice_num == 0 or choice_num > len(known_users):
            print(f"  Cancelled.")
            return False

        correct_user = known_users[choice_num - 1]

        # Save the correction for smart_finetune.py
        corrections_dir = os.path.join(LOG_DIR, "admin_corrections")
        os.makedirs(corrections_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            "face_embedding": face_embedding.cpu(),
            "voice_embedding": voice_embedding.cpu(),
            "correct_label": correct_user,
            "original_prediction": decision["predicted_user"],
            "original_confidence": decision["confidence"],
            "timestamp": timestamp
        }, os.path.join(corrections_dir, f"correction_{timestamp}.pt"))

        print(f"  ✓ Correction saved: this was {correct_user.upper()}")
        print(f"    Run 'python app/smart_finetune.py' to retrain with corrections.")
        return True

    except (ValueError, EOFError):
        print(f"  Invalid input. Skipping.")
        return False


# ============================================================
#  SECTION 12: MAIN AUTHENTICATION LOOP
# ============================================================

def run_single_authentication(models):
    """
    Execute one full authentication cycle using CaptureSession.

    The camera opens once and stays open for both password and face phases,
    giving the user a continuous live preview experience.

    Flow:
      1. Open CaptureSession (camera + preview window)
      2. Password phase: record audio with voice bar visible
      3. Verify password (Google STT + fuzzy match)
      4. Face capture phase: multi-frame collection with quality scoring
      5. Select best face → compute face embedding
      6. Liveness check: blink detection + head pose variation
      7. Compute voice embedding (reuses password audio)
      8. Run fusion model → make access decision
      9. Log attempt + offer admin override if rejected
     10. Close CaptureSession

    Args:
        models: ModelManager with all loaded models.

    Returns:
        dict: The final access decision.
    """
    print("\n" + "=" * 60)
    print("  🔐 AUTHENTICATION ATTEMPT")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Create capture session (opens camera + preview window)
    try:
        session = CaptureSession(models, show_preview=SHOW_PREVIEW)
    except RuntimeError as e:
        print(f"\n    ✗ {e}")
        return {
            "access_granted": False,
            "predicted_user": "unknown",
            "confidence": 0.0,
            "decision_path": "camera_failed",
            "details": {}
        }

    try:
        # ──────────────────────────────────────────────────
        # GATE 1: Voice Password Verification
        # ──────────────────────────────────────────────────
        print(f"\n  ────── GATE 1: VOICE PASSWORD ──────")
        print(f'  Please say the password: "{VOICE_PASSWORD}"')

        password_audio = None
        password_result = {"passed": False, "transcript": "", "method": "rejected", "score": 0.0}

        for attempt in range(MAX_PASSWORD_ATTEMPTS):
            if session.cancelled:
                break

            if attempt > 0:
                remaining = MAX_PASSWORD_ATTEMPTS - attempt
                print(f"\n    Attempt {attempt + 1}/{MAX_PASSWORD_ATTEMPTS} ({remaining} remaining)")

            # Record with live preview + voice bar
            password_audio = session.run_password_phase(PASSWORD_RECORD_DURATION)

            if password_audio is None:
                if session.cancelled:
                    break
                print(f"    ✗ Recording failed. Trying again...")
                continue

            password_result = verify_password(password_audio)

            if password_result["passed"]:
                break
            elif attempt < MAX_PASSWORD_ATTEMPTS - 1:
                print(f"    Try again — speak clearly and closer to the microphone.")

        # Handle cancellation
        if session.cancelled:
            cancelled_decision = {
                "access_granted": False,
                "predicted_user": "unknown",
                "confidence": 0.0,
                "decision_path": "cancelled_by_user",
                "details": {}
            }
            log_attempt(cancelled_decision, password_result)
            return cancelled_decision

        if not password_result["passed"]:
            print(f"\n  🔴 PASSWORD VERIFICATION FAILED after {MAX_PASSWORD_ATTEMPTS} attempts")
            print(f"     ACCESS DENIED — biometric scan skipped.")
            failed_decision = {
                "access_granted": False,
                "predicted_user": "unknown",
                "confidence": 0.0,
                "decision_path": "password_failed",
                "details": {}
            }
            log_attempt(failed_decision, password_result)
            return failed_decision

        # ──────────────────────────────────────────────────
        # GATE 2: BIOMETRIC SCAN (Multi-Frame Face Capture)
        # ──────────────────────────────────────────────────
        print(f"\n  ────── GATE 2: BIOMETRIC SCAN ──────")

        # Step 2a: Multi-frame face capture with quality scoring
        face_embedding = None
        raw_frame = None
        liveness_candidates = []

        for retry in range(MAX_FACE_RETRIES):
            if session.cancelled:
                break

            if retry > 0:
                print(f"\n    Face detection retry {retry + 1}/{MAX_FACE_RETRIES}...")

            candidates = session.run_face_capture_phase()

            if candidates and not session.cancelled:
                face_embedding, raw_frame = session.get_best_face_embedding(candidates)
                if face_embedding is not None:
                    liveness_candidates = candidates
                    break

        if session.cancelled:
            cancelled_decision = {
                "access_granted": False,
                "predicted_user": "unknown",
                "confidence": 0.0,
                "decision_path": "cancelled_by_user",
                "details": {}
            }
            log_attempt(cancelled_decision, password_result)
            return cancelled_decision

        if face_embedding is None:
            print(f"\n  🔴 FACE DETECTION FAILED after {MAX_FACE_RETRIES} attempts")
            print(f"     ACCESS DENIED")
            failed_decision = {
                "access_granted": False,
                "predicted_user": "unknown",
                "confidence": 0.0,
                "decision_path": "face_detection_failed",
                "details": {}
            }
            log_attempt(failed_decision, password_result)
            return failed_decision

        # Step 2b: Capture voice embedding
        # OPTIMIZATION: Reuse the password recording for the voice embedding.
        # This avoids making the user speak twice, which improves UX.
        # The ECAPA-TDNN captures speaker identity from ANY speech — the content
        # of what's being said doesn't matter, only the voice characteristics.

        # ──────────────────────────────────────────────────
        # LIVENESS CHECK (Anti-Spoofing Gate)
        # ──────────────────────────────────────────────────
        # Runs AFTER face capture, BEFORE fusion model.
        # Analyzes captured frames for blink detection + head pose variation.
        # If liveness fails → DENY immediately (skip voice embedding + fusion model).
        print(f"\n  ────── LIVENESS CHECK ──────")

        liveness_result = check_liveness(liveness_candidates, models)

        if not liveness_result["passed"]:
            print(f"\n  🔴 LIVENESS CHECK FAILED — ACCESS DENIED")
            print(f"     The system could not verify a live person is present.")
            liveness_decision = {
                "access_granted": False,
                "predicted_user": "unknown",
                "confidence": 0.0,
                "decision_path": f"liveness_failed ({liveness_result['method']})",
                "details": {"liveness": liveness_result}
            }
            log_attempt(liveness_decision, password_result, face_embedding, None)
            return liveness_decision

        # ──────────────────────────────────────────────────
        # VOICE EMBEDDING
        # ──────────────────────────────────────────────────
        print(f"\n    🔊 Computing voice embedding from password recording...")
        voice_embedding, _ = capture_voice_embedding(models, audio_data=password_audio)

        if voice_embedding is None:
            # Fallback: record fresh audio
            print(f"    ⚠ Reuse failed. Recording fresh voice sample...")
            voice_embedding, _ = capture_voice_embedding(models, audio_data=None)

        if voice_embedding is None:
            print(f"\n  🔴 VOICE EMBEDDING FAILED")
            print(f"     ACCESS DENIED")
            failed_decision = {
                "access_granted": False,
                "predicted_user": "unknown",
                "confidence": 0.0,
                "decision_path": "voice_embedding_failed",
                "details": {}
            }
            log_attempt(failed_decision, password_result)
            return failed_decision

        # ──────────────────────────────────────────────────
        # GATE 3: FUSION MODEL DECISION
        # ──────────────────────────────────────────────────
        decision = make_access_decision(models, face_embedding, voice_embedding)

        # Attach liveness result to the decision details
        decision["details"]["liveness"] = liveness_result

        # ──────────────────────────────────────────────────
        # LOG & ADMIN
        # ──────────────────────────────────────────────────
        log_attempt(decision, password_result, face_embedding, voice_embedding)

        # If rejected, offer admin override opportunity
        if not decision["access_granted"]:
            handle_admin_override(decision, face_embedding, voice_embedding)

        return decision

    finally:
        # Always close the session — releases camera + preview window
        session.close()


def display_final_result(decision):
    """Display the final access decision in a clear, visual format."""
    print("\n" + "=" * 60)

    if decision["access_granted"]:
        user = decision["predicted_user"].upper()
        conf = decision["confidence"]
        path = decision["decision_path"]

        print(f"  ╔══════════════════════════════════════════════════════╗")
        print(f"  ║              🟢 ACCESS GRANTED 🟢                   ║")
        print(f"  ╠══════════════════════════════════════════════════════╣")
        print(f"  ║  User:       {user:<40} ║")
        print(f"  ║  Confidence: {conf:<40.1%} ║")
        print(f"  ║  Method:     {path[:40]:<40} ║")
        print(f"  ╚══════════════════════════════════════════════════════╝")
    else:
        path = decision["decision_path"]

        print(f"  ╔══════════════════════════════════════════════════════╗")
        print(f"  ║              🔴 ACCESS DENIED 🔴                    ║")
        print(f"  ╠══════════════════════════════════════════════════════╣")
        print(f"  ║  Reason:     {path[:40]:<40} ║")
        print(f"  ╚══════════════════════════════════════════════════════╝")

    print("=" * 60)


# ============================================================
#  SECTION 13: MAIN ENTRY POINT
# ============================================================

def main():
    """
    Main entry point. Runs the authentication system in a continuous loop.
    Press Ctrl+C to exit gracefully.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  SECURE GATEWAY — Multi-Modal Biometric Security System  " + "║")
    print("║" + "  Face Recognition + Speaker Verification + Voice Password " + "║")
    print("╚" + "═" * 58 + "╝")

    # Step 1: Validate system
    if not validate_system():
        print("\n  ✗ System validation failed. Exiting.")
        sys.exit(1)

    # Step 2: Load all models
    models = ModelManager()
    try:
        models.load_all()
    except Exception as e:
        print(f"\n  ✗ Failed to load models: {e}")
        print(f"    Make sure you've run:")
        print(f"      1. python data_preparation/enroll_users.py")
        print(f"      2. python training/train_model.py")
        sys.exit(1)

    # Step 3: Authentication loop
    print("\n  System is LIVE. Waiting for authentication attempts.")
    print("  Press Ctrl+C at any time to shut down.\n")

    attempt_count = 0

    try:
        while True:
            attempt_count += 1

            # Wait for user to be ready
            input(f"\n  ▶ Press ENTER to start authentication attempt #{attempt_count}...")

            # Run one authentication cycle
            decision = run_single_authentication(models)

            # Display result
            display_final_result(decision)

            # Brief pause before next attempt
            print(f"\n  Ready for next attempt. Press ENTER when ready, or Ctrl+C to exit.")

    except KeyboardInterrupt:
        print(f"\n\n  System shutting down gracefully...")
        print(f"  Total attempts this session: {attempt_count}")

        # --- Release all resources ---
        # 1. Release camera (in case Ctrl+C hit during face capture)
        try:
            cv2 = _import_cv2()
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 2. Stop any active audio streams
        try:
            sd = _import_sounddevice()
            sd.stop()
        except Exception:
            pass

        # 3. Unload AI models and free GPU memory
        models.cleanup()

        # 4. Show log location
        if LOG_ATTEMPTS:
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(LOG_DIR, f"attempts_{date_str}.jsonl")
            if os.path.exists(log_file):
                print(f"  Logs saved to: {log_file}")

        print(f"  All resources released. Goodbye!\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    main()
