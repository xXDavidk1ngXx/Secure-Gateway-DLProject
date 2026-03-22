# Secure Gateway — Project Setup Guide

A step-by-step guide for cloning, installing, and running the Secure Gateway biometric authentication system.

---

## What's Included in the GitHub Repo

The repo already contains **everything needed to skip straight to evaluation and the live demo** without re-running the data pipeline or training. Specifically, the repo includes:

- All trained model weights (`models/fusion_model.pt`, `models/user_profiles.pt`)
- Pre-computed embeddings (`data/embeddings/face_embeddings.pt`, `voice_embeddings.pt`)
- Training history & data splits (`models/training_history.pt`, `models/data_splits.pt`)
- ECAPA-TDNN pretrained model cache (`models/voice/ecapa_tdnn/`)
- MediaPipe liveness model (`models/face_landmarker.task`)
- All source code, config, and evaluation scripts

**Not included** (gitignored): raw face photos, raw voice recordings, processed/augmented data, and the virtual environment.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or higher (tested on 3.11) |
| **Git** | Any recent version |
| **NVIDIA GPU + CUDA** | Recommended for speed (CUDA 12.1); CPU works but is slower |
| **Webcam** | Required for the live authentication demo |
| **Microphone** | Required for voice recording in the live demo |
| **Internet connection** | Required for Google Speech-to-Text API (password verification) |
| **Operating System** | Windows 10/11 (tested), Linux/Mac should also work |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/xXDavidk1ngXx/Secure-Gateway-DLProject.git
cd Secure-Gateway-DLProject
```

## Step 2 — Create a Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3 — Install PyTorch (with CUDA support)

This must be installed **before** the other dependencies because SpeechBrain and facenet-pytorch depend on it.

**With NVIDIA GPU (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (no NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio
```

## Step 4 — Install All Other Dependencies

```bash
pip install -r requirements.txt
```

This installs: `facenet-pytorch`, `speechbrain`, `SpeechRecognition`, `sounddevice`, `mediapipe`, `opencv-python`, `scikit-learn`, `matplotlib`, `scipy`, `numpy`, `Pillow`, `tqdm`.

> **Note:** If `pip install -r requirements.txt` fails on the PyTorch lines (because they have `+cu121` version tags), you can safely ignore those errors — you already installed PyTorch in Step 3. Alternatively, install the remaining packages manually:
> ```bash
> pip install facenet-pytorch speechbrain SpeechRecognition sounddevice mediapipe opencv-python scikit-learn matplotlib scipy numpy Pillow tqdm
> ```

## Step 5 — Verify the Setup

Run the config printer to verify everything loaded correctly:
```bash
python utils/config.py
```

This should print all configuration values without errors, and show whether GPU or CPU is detected.

---

## What Can Be Run (and in What Order)

### Option A — Quick Verification (No raw data needed)

Since the repo includes pre-computed embeddings and trained models, you can **skip the entire data pipeline** and go directly to evaluation and the live demo.

#### A1. Generate Evaluation Plots (no webcam/mic needed)

```bash
python evaluation/visualizer.py
```

This produces 7 visualization plots in `evaluation/figures/`:
- t-SNE embedding clusters
- Training curves (loss & accuracy)
- Confusion matrix (99.5% test accuracy)
- Genuine vs. impostor similarity distributions
- Per-class performance bars
- Architecture diagram
- Combined dashboard

#### A2. Run the Live Authentication System (requires webcam + mic + internet)

```bash
python app/run_system.py
```

This launches the full real-time system:
1. Asks for the spoken password ("my voice is my password")
2. Opens the webcam for face capture with live preview
3. Runs liveness detection (blink + head pose)
4. Runs the fusion model for identity classification
5. Grants or denies access

> **Note:** Since the lecturer is not an enrolled user (David, Itzhak, or Yossi), the system should correctly classify them as **"unknown"** and deny access. This is the expected behavior and demonstrates that the system works.

Press `Ctrl+C` to exit the live system gracefully.

---

### Option B — Full Pipeline from Scratch (requires raw data)

If you want to reproduce the entire pipeline from data processing through training:

#### B0. Obtain the Raw Dataset

The raw face photos and voice recordings are **not in the repo** (privacy). They are available in the project's Google Drive folder (link provided in the submitted docs). Download and place them as:
```
data/face/raw/david/      <- David's face photos
data/face/raw/itzhak/     <- Itzhak's face photos
data/face/raw/yossi/      <- Yossi's face photos
data/voice/raw/david/     <- David's voice recordings
data/voice/raw/itzhak/    <- Itzhak's voice recordings
data/voice/raw/yossi/     <- Yossi's voice recordings
```

#### B1. Preprocess faces — MTCNN detection, alignment, crop to 160x160

```bash
python data_preparation/preprocess_faces.py
```

#### B2. Preprocess voices — resample to 16kHz, trim silence, normalize

```bash
python data_preparation/preprocess_voices.py
```

#### B3. Augment faces — 15 augmentations per image (~480/person)

```bash
python data_preparation/augment_face.py
```

#### B4. Augment voices — 3 augmentations per clip (~120/person)

```bash
python data_preparation/augment_voice.py
```

#### B5. Compute embeddings — FaceNet (512-d) + ECAPA-TDNN (192-d)

```bash
python data_preparation/compute_embeddings.py
```

#### B6. Enroll users — compute mean reference profiles

```bash
python data_preparation/enroll_users.py
```

#### B7. Train the fusion model — Late Fusion MLP (704 -> 4 classes)

```bash
python training/train_model.py
```

#### B8. Generate evaluation plots

```bash
python evaluation/visualizer.py
```

#### B9. Run the live system

```bash
python app/run_system.py
```

---

## Summary Checklist

| Step | Command | Requires Raw Data? | Requires Hardware? |
|------|---------|-------------------|--------------------|
| Clone repo | `git clone ...` | No | No |
| Create venv | `python -m venv venv` | No | No |
| Install PyTorch | `pip install torch ...` | No | No |
| Install deps | `pip install -r requirements.txt` | No | No |
| Verify config | `python utils/config.py` | No | No |
| **Evaluation plots** | `python evaluation/visualizer.py` | **No** | No |
| **Live demo** | `python app/run_system.py` | **No** | Webcam + Mic + Internet |
| Full pipeline (B1-B9) | See above | **Yes** | GPU recommended |

The project can be verified by running just **Steps 1-5 + evaluation plots** (no hardware needed), and optionally the **live demo** if webcam and microphone are available.
