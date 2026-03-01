# Secure Gateway — Multi-Modal Biometric Authentication System

A real-time access control system that authenticates users through **face recognition**, **speaker verification**, and a **spoken password** — all fused into a single decision using deep learning.

Built as a Deep Learning capstone project. The system grants access exclusively to three enrolled team members and rejects all others, including spoof attempts.

in addition ot the README.md, for more in-depth information you can check the project_sumamry.md file.

---

## Authors

David Khutsishvili, Itzhak Mutzeri, Yossi Yadgar

Deep Learning — Final Project

---
## How It Works

```
User → Gate 1: Voice Password (Speech-to-Text + fuzzy match)
         ↓ pass
       Gate 2: Biometric Capture
         ├── Face  → MTCNN → FaceNet → 512-dim embedding
         └── Voice → ECAPA-TDNN → 192-dim embedding
         ↓
       Liveness Check (blink detection + head pose)
         ↓ pass
       Gate 3: Fusion Model (704-dim MLP → identity classification)
         ↓
       3-Tier Decision → ACCESS GRANTED / DENIED
```

Each gate can reject independently. An attacker must defeat **all channels simultaneously** to gain access. Failure at any gate terminates the attempt early — if the password is wrong, biometric capture never starts.

### Decision Logic
- **>= 85% confidence** → Immediate access
- **50–85%** → Gray area: cosine similarity fallback against enrollment profiles (both face >= 0.4 and voice >= 0.4 must pass)
- **< 50% or "unknown"** → Access denied

---

## Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended; CPU works but slower)
- Webcam + Microphone
- Internet connection (for Google Speech-to-Text)
- The Data set (not mendatory to work): you can put your own dataset. 
our datasetis in the Drive project folder, link in the docs file sent, needs to be downloaded separately. 
once the raw data is downloaded, put it in their respective categories (face and voice) and you can start running the project from start to finish.

### Installation

```bash
git clone https://github.com/xXDavidk1ngXx/secure-gateway-DLProject.git
cd secure-gateway-DLProject
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install facenet-pytorch speechbrain opencv-python matplotlib scikit-learn Pillow
pip install SpeechRecognition sounddevice scipy mediapipe tqdm numpy
```

**Or**, install all dependencies at once from the requirements file:

```bash
git clone https://github.com/xXDavidk1ngXx/secure-gateway-DLProject.git
cd secure-gateway-DLProject
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Pipeline — Run in Order

```bash
# Phase 1: Data preparation (run once if data is available)
python data_preparation/preprocess_faces.py
python data_preparation/preprocess_voices.py
python data_preparation/augment_face.py
python data_preparation/augment_voice.py
python data_preparation/compute_embeddings.py
python data_preparation/enroll_users.py

# Phase 2: Train the fusion model
python training/train_model.py

# Phase 3: Generate evaluation plots
python evaluation/visualizer.py

# Phase 4: Run the live system
python app/run_system.py
```

---

## Features

### Authentication Pipeline

- **Face Recognition** — MTCNN detects and aligns faces from the webcam, FaceNet (InceptionResnetV1, pretrained on VGGFace2) extracts a 512-dimensional identity embedding.
- **Speaker Verification** — ECAPA-TDNN (pretrained on VoxCeleb, 7000+ speakers) extracts a 192-dimensional voice identity embedding from the microphone recording.
- **Spoken Password** — Google Speech Recognition API transcribes the user's speech, then fuzzy string matching (SequenceMatcher, 75% threshold) verifies the password phrase. A keyword fallback catches imperfect transcriptions.
- **Late Fusion MLP** — Face (512d) and voice (192d) embeddings are concatenated into a 704-dimensional vector and classified by a trained neural network into 4 classes: david, itzhak, yossi, or unknown.
- **3-Tier Confidence Decision** — High confidence grants immediately, gray area falls back to cosine similarity against enrolled profiles, low confidence denies.
- **Audio Reuse** — The password recording is reused for voice embedding extraction, so the user only speaks once.

### Security

- **Multi-Gate Architecture** — Three independent gates (password, biometrics, fusion), each capable of rejecting on its own. Defense in depth.
- **Liveness Detection** — Blink detection (Eye Aspect Ratio via MediaPipe Face Mesh) and head pose variation (solvePnP) run after face capture. Blocks photo prints and video replay attacks. OR logic: either check passing is sufficient.
- **Cross-Modal Spoofing Detection** — Mismatched face+voice pairs (e.g., David's face + Yossi's voice) activate the "unknown" class. The model was trained on cross-person pairs specifically for this.
- **Lockout Protection** — After 3 failed password attempts, the system locks out for 30 seconds to prevent brute-force attacks.

### Live System UX

- **Live Camera Preview** — OpenCV window displays real-time bounding boxes with color-coded quality feedback: green (good detection), orange (marginal), red (no face).
- **Multi-Frame Quality Scoring** — Captures ~3 seconds of video after the first face detection, scores every frame on detection confidence, face size, and centering, then selects the best frame for embedding.
- **Graceful Shutdown** — Ctrl+C cleanly releases the camera, stops microphone streams, unloads models from GPU memory, and reports log file locations.
- **GPU Auto-Detection** — Automatically uses CUDA if available, falls back to CPU seamlessly.

### Data Pipeline

- **Face Preprocessing** — MTCNN detection, eye-based alignment, margin cropping (40px), resize to 160x160 pixels. Strict 0.95 confidence threshold filters ambiguous detections.
- **Voice Preprocessing** — Resampling to 16kHz mono, silence trimming (-40dB threshold with 0.15s padding), peak normalization to -1dB, and quality validation (rejects clips <0.5s, >30s, corrupted, or near-silent).
- **Face Augmentation (15x)** — Horizontal flip, rotation (+-15 degrees), brightness/contrast jitter, Gaussian blur, random erasing (cutout), Gaussian noise, perspective transform. Expands ~30 images to ~480 per person.
- **Voice Augmentation (3x)** — Additive background noise, pitch shifting (+-2 semitones), speed perturbation (0.9x-1.1x). Expands ~30 clips to ~120 per person.
- **Embedding Computation** — Batch extraction through frozen pretrained models (FaceNet for face, ECAPA-TDNN for voice). Transfer learning — no fine-tuning required.
- **User Enrollment** — Computes mean embedding per person per modality, L2-normalized onto the unit hypersphere. Includes per-user quality assessment (consistency check against the centroid).

### Training

- **Leakage-Free Data Splitting** — Train (70%) / validation (15%) / test (15%) splits happen per-person, per-modality, *before* creating face-voice pairs. No embedding appears in more than one set.
- **Genuine + Unknown Pair Generation** — Same-person face+voice = genuine pair. Cross-person face+voice = unknown pair. All 6 cross-person combinations used evenly, balanced ~50/50 against genuine pairs.
- **BatchNorm** — Essential because voice embeddings have ~800x larger variance than face embeddings. Without it, the 192 voice dimensions dominate the 512 face dimensions.
- **Class-Weighted Loss** — Inverse-frequency weighting compensates for unequal sample counts across users.
- **Early Stopping + Checkpointing** — Training halts if validation loss doesn't improve for 20 epochs. Best model saved by minimum validation loss (not accuracy).
- **LR Scheduling** — ReduceLROnPlateau reduces learning rate when validation loss plateaus.
- **Full Reproducibility** — All RNGs seeded (Python, NumPy, PyTorch CPU/CUDA), CuDNN deterministic mode enabled.

### Monitoring

- **Comprehensive Logging** — Every authentication attempt logged to daily JSONL files with timestamp, decision, confidence, transcript, predicted user, cosine similarities, liveness results, and decision path.
- **Admin Override** — After a false rejection, an admin can press a key to record the correction, saving the embeddings with the correct label for potential future retraining.
- **Centralized Configuration** — A single `utils/config.py` with 15 class-based namespaces governs all paths, thresholds, and hyperparameters across every script. Change a value once, it updates everywhere.

---

## Tech Stack

| Component | Technology | Output |
|---|---|---|
| Face Detection | MTCNN (facenet-pytorch) | Bounding box + 5 landmarks |
| Face Embedding | InceptionResnetV1 (pretrained on VGGFace2, 3.3M faces) | 512-dim vector |
| Voice Embedding | ECAPA-TDNN (SpeechBrain, pretrained on VoxCeleb, 7000+ speakers) | 192-dim vector |
| Speech-to-Text | Google Speech Recognition API | Text transcription |
| Liveness Detection | MediaPipe FaceLandmarker (EAR + solvePnP) | Blink count + head pose |
| Fusion Model | Custom MLP (PyTorch): 704 → 256 → 128 → 4 classes | Identity + confidence |
| Camera / Audio | OpenCV, sounddevice, scipy | Frames / waveforms |
| Visualization | matplotlib, scikit-learn (t-SNE) | 7 evaluation plots |

### Fusion MLP Architecture
```
Input(704) → Linear(704→256) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(256→128) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(128→4)   → [david, itzhak, yossi, unknown]
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.5%** (376/378) |
| Per-User Accuracy | David 100%, Itzhak 100%, Yossi 100% |
| Unknown Detection | 98% recall, 100% precision |
| Embedding Separation | Clear t-SNE clusters, minimal overlap |

---

## Visualizations

The evaluation suite (`evaluation/visualizer.py`) generates 7 plots saved to `evaluation/figures/` at 300 DPI:

| Plot | What It Proves |
|---|---|
| t-SNE Clusters | Fused 704-dim embeddings are well-separated per person |
| Training Curves | Model converges without overfitting (train/val track closely) |
| Confusion Matrix | 99.5% accuracy, zero inter-user confusion |
| Similarity Distributions | Genuine vs. impostor cosine scores are clearly separated |
| Per-Class Performance | All users achieve high precision and recall |
| Architecture Diagram | Full multi-gate pipeline overview |
| Dashboard | Combined view of all 6 plots |

---

## Project Structure

```
secure-gateway-DLProject/
├── app/
│   └── run_system.py               # Main executable — live authentication
├── data/
│   ├── embeddings/                  # Face & voice .pt embedding files
│   ├── face/ (raw/processed/augmented)
│   └── voice/ (raw/processed/augmented)
├── data_preparation/
│   ├── preprocess_faces.py          # MTCNN detect + align + crop
│   ├── preprocess_voices.py         # Resample + trim + normalize
│   ├── augment_face.py              # 15 augmentations per image
│   ├── augment_voice.py             # 3 augmentations per clip
│   ├── compute_embeddings.py        # FaceNet + ECAPA-TDNN extraction
│   └── enroll_users.py              # Mean profile enrollment
├── training/
│   └── train_model.py               # Fusion MLP training pipeline
├── evaluation/
│   ├── visualizer.py                # 7 evaluation plots + dashboard
│   └── figures/                     # Generated PNGs (300 DPI)
├── models/
│   ├── fusion_model.pt              # Trained fusion MLP checkpoint
│   ├── user_profiles.pt             # Enrolled reference embeddings
│   ├── data_splits.pt               # Train/val/test split indices
│   ├── training_history.pt          # Per-epoch loss/accuracy curves
│   ├── face_landmarker.task         # MediaPipe liveness model
│   └── voice/ecapa_tdnn/            # Pretrained ECAPA-TDNN cache
├── utils/
│   └── config.py                    # Centralized configuration (15 namespaces)
├── logs/                            # Authentication attempt logs (JSONL)
└── requirements.txt                 # Pinned Python dependencies
```

---

