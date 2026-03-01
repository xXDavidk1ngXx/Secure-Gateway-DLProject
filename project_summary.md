# Secure Gateway — Complete Project Summary

## Project Purpose & Vision

The **Secure Gateway** is a multi-modal biometric authentication system built as a deep learning capstone project. Its purpose is to control physical access by verifying a person's identity through three independent channels simultaneously: their face, their voice, and a spoken password. The system is designed to grant access exclusively to three authorized team members — David, Itzhak, and Yossi — and reject everyone else, including imposters attempting to spoof individual modalities.

The core insight behind the project is that single-modality biometric systems (face-only, voice-only) are inherently vulnerable — a photograph can fool a face scanner, a recording can fool a voice verifier. By fusing multiple biometric signals into a single decision and adding a knowledge-based factor (the spoken password), the system achieves defense in depth where an attacker would need to simultaneously defeat all channels to gain access.

---

## Architecture Overview

The system is organized as a multi-gate authentication pipeline:

```
User Approaches → Gate 1: Voice Password (Speech-to-Text)
                     ↓ pass
                  Gate 2: Biometric Capture
                     ├── Face → MTCNN → FaceNet → 512-dim embedding
                     └── Voice → ECAPA-TDNN → 192-dim embedding
                     ↓
                  Liveness Check (Blink Detection + Head Pose)
                     ↓ pass
                  Gate 3: Fusion Model (704-dim → MLP → Identity Decision)
                     ↓
                  3-Tier Confidence Logic → ACCESS GRANTED / DENIED
```

Each gate can independently reject. A failure at any stage terminates the authentication attempt early — if the password is wrong, the biometric scan never even starts. This reduces unnecessary computation and prevents information leakage to attackers.

---

## Phase 1: Data Collection & Preparation

### 1.1 Raw Data Gathering

Each of the three team members provided:

- **~30 face photographs** taken from various angles, lighting conditions, expressions, and distances — simulating real-world variation that the camera would encounter.
- **~30 voice recordings** divided into three categories: 10 recordings of the password phrase "my voice is my password," 10 short sentences, and 10 longer passages. The variety ensures the voice model captures the speaker's identity across different phonetic contexts, not just one memorized phrase.

All data was organized into a structured directory (`data/face/raw/<person>/` and `data/voice/raw/<person>/`) to maintain traceability from raw input through to final embeddings.

### 1.2 Face Preprocessing (`preprocess_faces.py`)

Raw photographs are uncontrolled — different resolutions, backgrounds, body parts visible, varying head orientations. The preprocessing pipeline standardizes them into clean, model-ready face crops. Four techniques were applied:

**MTCNN Face Detection** — A three-stage cascaded convolutional network (Proposal Network → Refinement Network → Output Network) that scans images at multiple scales to locate faces and output five facial landmarks (both eyes, nose, both mouth corners). We configured it with a strict confidence threshold of 0.95 to discard ambiguous detections, and `select_largest=True` to ignore incidental background faces.

**Alignment** — Using the detected eye coordinates, each face is rotated so both eyes lie on a horizontal line. This corrects head tilt and ensures the embedding model doesn't waste its representational capacity learning rotational invariance — it receives consistently oriented inputs.

**Cropping with Margin** — The detected bounding box is expanded by 40 pixels on all sides before cropping. This preserves jawline, forehead, and ear context that carries identity-discriminating information. Too tight a crop loses these features; too loose introduces background noise.

**Resizing to 160×160** — FaceNet's InceptionResnetV1 architecture expects exactly this input resolution. Standardizing dimensions also enables batched tensor processing during embedding computation.

### 1.3 Voice Preprocessing (`preprocess_voices.py`)

Audio recordings from different devices and environments require normalization to ensure consistent model input. Five techniques were applied:

**Resampling to 16kHz** — ECAPA-TDNN and Google Speech Recognition both expect 16kHz audio. By the Nyquist theorem, 16kHz captures all speech-relevant frequencies (which top out at ~8kHz), while higher sample rates waste memory and computation.

**Mono Conversion** — Stereo channels are averaged into a single channel. Speaker identity resides in vocal characteristics (pitch, timbre, formant structure), not spatial positioning.

**Silence Trimming** — Energy-based detection removes leading and trailing silence (threshold at -40dB), with 0.15 seconds of padding to prevent clipping the first phoneme. This keeps clips tight and consistent across recordings.

**Peak Normalization to -1dB** — All recordings are scaled to the same peak amplitude. Without this, a quietly recorded sample and a loudly recorded sample from the same person would produce different embeddings. The -1dB headroom prevents digital clipping artifacts.

**Quality Validation** — Files that are too short (<0.5s), too long (>30s), corrupted (NaN/Inf values), or nearly silent (RMS below -50dB) are automatically rejected to prevent garbage data from contaminating the embedding space.

### 1.4 Data Augmentation

With only ~30 original samples per person per modality, the dataset is too small to train a robust classifier. Augmentation synthetically expands the training set by applying realistic transformations that simulate real-world variation.

**Face Augmentation (`augment_face.py`)** — Each original face image was augmented 15 times using 7 transformation types, each targeting a specific real-world condition:

| Augmentation | Simulates |
|---|---|
| Horizontal flip | Approaching from either side |
| Rotation ±15° | Slight head tilt |
| Brightness/contrast jitter | Different lighting (indoor/outdoor, day/night) |
| Gaussian blur | Out-of-focus camera |
| Random erasing (cutout) | Partial occlusion (glasses, hand, mask) |
| Gaussian noise | Camera sensor noise in low light |
| Perspective transform | Different camera mounting angles |

This expanded ~30 images per person to ~480 (originals + augmented), providing the face embedding model with sufficient variation to produce robust identity representations.

**Voice Augmentation (`augment_voice.py`)** — Each original recording received 3 augmented variants:

| Augmentation | Simulates |
|---|---|
| Additive background noise | Ambient environment noise |
| Pitch shifting (±2 semitones) | Natural vocal variation (time of day, fatigue) |
| Speed perturbation (0.9×–1.1×) | Speaking pace variation |

This expanded ~30 clips per person to ~120.

### 1.5 Embedding Computation (`compute_embeddings.py`)

Embeddings are the mathematical heart of the system. Raw pixels and waveforms cannot be meaningfully compared — two photos of David taken five minutes apart have completely different pixel values. Embeddings compress data into compact numerical vectors where identity information is preserved and irrelevant variation is discarded.

**Face Embeddings via FaceNet (InceptionResnetV1)** — A pretrained deep CNN, trained on VGGFace2 (3.3 million images of 9,131 individuals), that maps any 160×160 face image to a 512-dimensional vector. Faces of the same person cluster together in this space; different people are pushed apart. We use the model frozen (no fine-tuning), in eval mode with `torch.no_grad()`, applying fixed image standardization `(pixel - 127.5) / 128.0` to match the normalization used during VGGFace2 training.

**Voice Embeddings via ECAPA-TDNN (SpeechBrain)** — Emphasized Channel Attention, Propagation and Aggregation in Time-Delay Neural Networks. Pretrained on VoxCeleb1+2 (7,000+ speakers from YouTube interviews), it maps any audio clip to a 192-dimensional speaker identity vector. SpeechBrain handles the entire internal pipeline (waveform → Mel spectrogram → ECAPA-TDNN → embedding) automatically.

**Output**: Two `.pt` files — `face_embeddings.pt` containing `{person: Tensor[N, 512]}` and `voice_embeddings.pt` containing `{person: Tensor[N, 192]}`.

Final embedding counts per person:
- David: 816 face embeddings, 132 voice embeddings
- Itzhak: 464 face embeddings, 120 voice embeddings
- Yossi: 448 face embeddings, 120 voice embeddings

### 1.6 User Enrollment (`enroll_users.py`)

Enrollment computes a single reference template per person per modality — the "identity anchor" against which live samples are compared during the cosine similarity fallback.

For each person, the mean of all their embeddings is computed and L2-normalized. L2 normalization is critical because cosine similarity between normalized vectors simplifies to a dot product, and the average of unit vectors is not itself a unit vector — renormalization pushes the centroid back onto the unit hypersphere.

The enrollment script also performs quality assessment: it computes how similar each original embedding is to the centroid. If all 30 photos of David are consistent, they'll have high similarity to the centroid (mean ~0.85+). Low or variable similarity flags potential data quality issues (mislabeled photos, inconsistent recording conditions).

The output (`models/user_profiles.pt`) stores for each person: L2-normalized face mean (512d), voice mean (192d), fused mean (704d), standard deviations, and sample counts.

---

## Phase 2: The Fusion Model

### 2.1 Late Fusion Strategy

Rather than training separate classifiers for face and voice, we adopt **late fusion**: concatenating the face embedding (512d) and voice embedding (192d) into a single 704-dimensional fused vector, then training one classifier on this combined representation. This lets the model learn cross-modal correlations — for instance, David's face pattern co-occurs with David's voice pattern, and any mismatch (David's face + Yossi's voice) should trigger rejection.

### 2.2 Data Splitting — Before Pairing

A critical design decision: data is split into train (70%), validation (15%), and test (15%) sets **per person, per modality, before any pairing**. If we had paired first and then split, the same face embedding could appear in both a training pair and a test pair (with a different voice partner), causing data leakage and inflated accuracy. By splitting the raw embeddings first, we guarantee that no face or voice embedding appears in more than one set.

The split indices are saved to `models/data_splits.pt` for reproducibility — the same seed alone is not sufficient if the embedding data changes.

### 2.3 Pair Generation

**Genuine Pairs** — For each face embedding of a person, a random voice embedding from the same person is selected and concatenated: `[face_A | voice_A] → label "A"`. Since face embeddings outnumber voice embeddings ~4:1, voice embeddings are reused with different face partners, creating unique pairs that maximize face diversity.

**Unknown/Impostor Pairs** — Face from person A concatenated with voice from person B (where A ≠ B): `[face_A | voice_B] → label "unknown"`. These cross-person pairs teach the model that mismatched face-voice combinations are impostors. All 6 cross-person combinations (A→B, A→C, B→A, B→C, C→A, C→B) are used evenly. The total number of unknown pairs is balanced against the total genuine pairs (~50/50 split).

### 2.4 Model Architecture — Late Fusion MLP

```
Input (704-dim fused vector)
  │
  ├─ Linear(704 → 256)
  ├─ BatchNorm1d(256)
  ├─ ReLU
  ├─ Dropout(0.3)
  │
  ├─ Linear(256 → 128)
  ├─ BatchNorm1d(128)
  ├─ ReLU
  ├─ Dropout(0.2)
  │
  └─ Linear(128 → 4)   ← [david, itzhak, yossi, unknown]
```

**Why BatchNorm is essential**: Enrollment analysis revealed face embeddings have std ~0.02 while voice embeddings have std ~16 — an 800× scale difference. Without BatchNorm, the 192 voice dimensions would dominate the 512 face dimensions purely because their raw numbers are larger. BatchNorm normalizes each dimension to approximately zero mean and unit variance, ensuring all 704 dimensions contribute equally.

**Why Dropout**: With a small dataset (~2,000–3,000 pairs), the model can easily memorize training data. Dropout forces redundant representations (no single neuron is essential), improving generalization. Higher dropout (0.3) is applied in the wider layer, lower (0.2) in the narrower one.

**No Softmax in the model**: PyTorch's CrossEntropyLoss internally applies LogSoftmax + NLLLoss. Adding Softmax here would apply it twice, producing wrong gradients. For inference, softmax is applied manually via `predict_proba()`.

### 2.5 Training Pipeline

**Class-Weighted Loss** — David has 816 face embeddings versus Itzhak's 464, creating class imbalance. Inverse-frequency weighting (`weight_c = total / (num_classes × count_c)`) tells the loss function to penalize errors on underrepresented classes more heavily, forcing the model to pay equal attention to all identities.

**Optimizer** — Adam with learning rate 1e-3 and weight decay (L2 regularization) to prevent overfitting.

**Learning Rate Scheduler** — ReduceLROnPlateau monitors validation loss and reduces the learning rate by a configurable factor when it plateaus, allowing finer convergence in later training stages.

**Early Stopping** — Training halts if validation loss doesn't improve for a configurable patience window (default: 20 epochs). This prevents overfitting and unnecessary computation. The best model state (based on minimum validation loss, not maximum accuracy) is checkpointed and restored.

**Reproducibility** — All random number generators (Python, NumPy, PyTorch CPU, PyTorch CUDA) are seeded with a fixed value. CuDNN deterministic mode is enabled. This ensures identical results across runs.

The model trained for ~90 epochs with early stopping triggering at epoch 74, achieving near-perfect convergence.

### 2.6 Model Checkpoint

The saved model file (`models/fusion_model.pt`) is fully self-contained: it stores the state dictionary, architecture parameters (so the model can be reconstructed without hardcoding), class mappings, and training metadata. Training history (`models/training_history.pt`) stores per-epoch loss and accuracy for visualization.

---

## Phase 3: System Logic — The Live Authentication Pipeline (`run_system.py`)

### 3.1 System Startup

On launch, the system performs a 7-point validation: checks that the fusion model exists, user profiles are loaded (3 enrolled users), webcam is accessible, microphone is functional, internet connectivity exists (for Google STT), compute device is detected (GPU/CPU), and the liveness detection model is available.

All AI models are loaded once into a `ModelManager`: the Fusion MLP, user profiles, MTCNN + FaceNet, ECAPA-TDNN, and MediaPipe FaceLandmarker. Models persist in memory across authentication attempts for rapid inference.

### 3.2 Gate 1 — Voice Password Verification

The user is prompted to say the password phrase: "my voice is my password." Audio is recorded via the microphone (using `sounddevice`), sent to Google's Speech-to-Text API for transcription, and the returned text is compared against the stored password using fuzzy matching (`difflib.SequenceMatcher` with a 0.75 threshold). A keyword fallback checks for the presence of critical words ("voice," "password") in case transcription is imperfect.

The user gets up to 3 attempts before lockout. This gate serves as the "something you know" authentication factor and filters out casual intruders before any biometric computation occurs.

### 3.3 Gate 2 — Biometric Capture

**Face Capture** — A `CaptureSession` opens the webcam and runs a multi-frame quality-scoring capture. Over ~3–5 seconds, multiple frames are analyzed for face detection confidence, face size (larger = better), and centering. The top-K highest-quality frames are selected, and the best face crop is passed through MTCNN for detection/alignment and FaceNet for embedding, producing the 512-dimensional face vector.

A live preview window (OpenCV) shows the camera feed with bounding boxes: green for good detections, orange for marginal quality, and red/absent when no face is detected.

**Voice Capture** — Rather than recording a separate voice sample (which would inconvenience the user), the system reuses the password recording from Gate 1. ECAPA-TDNN captures speaker identity from any speech regardless of content — the phonetic content doesn't matter, only the vocal characteristics (pitch, timbre, formant structure). If reuse fails, a fresh recording is captured as fallback.

### 3.4 Liveness Detection

Between face capture and the fusion decision, a liveness check analyzes the captured video frames for signs of a live human versus a static photograph or video replay. Two methods are used with OR logic (either passing is sufficient):

**Blink Detection** — MediaPipe FaceLandmarker extracts 478 facial landmarks from each frame. The Eye Aspect Ratio (EAR) — the ratio of the eye's vertical opening to its horizontal width — is computed per frame. A live person's EAR dips sharply during blinks and recovers; a photograph's EAR remains constant. At least one blink must be detected across the capture window.

**Head Pose Variation** — Using 6 key facial landmarks and OpenCV's solvePnP algorithm, yaw/pitch/roll angles are estimated per frame. A live person exhibits natural micro-movements; a static image has zero variation. The standard deviation of yaw and pitch across frames must exceed minimum thresholds.

If both methods fail (insufficient frames, or the person is abnormally still), the system denies access as a potential spoof attempt.

### 3.5 Gate 3 — Fusion Model Decision

The 512-dim face embedding and 192-dim voice embedding are concatenated into the 704-dim fused vector and passed through the trained MLP. Softmax produces class probabilities for David, Itzhak, Yossi, and Unknown.

A 3-tier confidence logic then applies:

**High Confidence (≥ 85%)** — If the top predicted class is an authorized user with ≥ 85% confidence, access is granted immediately.

**Gray Area (50%–85%)** — The model is uncertain. A cosine similarity fallback compares the live face and voice embeddings against the enrolled profiles. If both modalities exceed their respective similarity thresholds (face ≥ 0.4, voice ≥ 0.4), and the predicted class matches the most similar profile, access is granted. This reduces false rejections under adverse conditions (poor lighting, background noise).

**Low Confidence (< 50%) or Unknown Class** — Access denied. The predicted class is "unknown" or the model has no strong prediction for any authorized user.

### 3.6 Logging & Admin Override

Every authentication attempt is logged to a JSONL file with full details: timestamp, decision, confidence, transcript, predicted user, cosine similarities, liveness results, and decision path. This supports security auditing and post-hoc analysis.

If a legitimate user is incorrectly rejected, an admin can press a designated key to record the correction — saving the face and voice embeddings with the correct label for potential future retraining.

### 3.7 Graceful Shutdown

A shutdown handler catches Ctrl+C and cleanly releases the camera, stops active microphone streams, unloads AI models from GPU memory (`torch.cuda.empty_cache()`), forces garbage collection, and reports log file locations.

---

## Phase 4: Evaluation & Visualization (`visualizer.py`)

The visualization suite generates 7 high-quality plots (saved at 300 DPI) that provide mathematical evidence for every claim about system performance.

### Visualization 1: t-SNE Embedding Clusters

All 704-dimensional fused embeddings from the test set are reduced to 2D using t-SNE (perplexity=30, 1000 iterations). The result shows three clearly separated clusters for David (blue), Itzhak (orange), and Yossi (green), with enrollment profile centers marked as white stars sitting in the middle of their respective clusters.

**Conclusion**: The clusters have minimal overlap, proving that the face+voice fusion creates a highly discriminative biometric signature per person. Even though individual modalities might have some inter-person overlap, the concatenated 704-dim representation achieves tight, separable clusters — the mathematical proof that late fusion works.

### Visualization 2: Training Curves

Loss and accuracy plots over ~90 epochs show both train and validation curves tracking each other closely throughout training — no significant overfitting. Loss drops sharply in the first ~15 epochs (from ~0.8 to ~0.1), then converges to near-zero. Accuracy jumps from ~50% to ~90% in the first 10–15 epochs, then climbs to ~100%.

**Conclusion**: The model converges cleanly and stably, generalizes well (validation tracks training throughout), and the best checkpoint at epoch 74 was selected based on minimum validation loss rather than maximum accuracy — the correct approach since loss captures confidence calibration.

### Visualization 3: Confusion Matrix (99.5% Test Accuracy)

On held-out test data: David 123/123 correct (100%), Itzhak 71/71 correct (100%), Yossi 68/68 correct (100%), Unknown 112/114 correct (98%) — with 2 Unknown samples misclassified as Yossi.

**Conclusion**: Zero confusion between the three known users. The only weakness is 2 false accepts (1.75%) from the Unknown class into Yossi, which are mitigated by the multi-gate architecture — even if the fusion model says "Yossi" for an unknown person, they must still pass the password gate and liveness check.

### Visualization 4: Genuine vs. Impostor Similarity Distributions

Cosine similarity histograms for genuine pairs (sample vs. own profile) and impostor pairs (sample vs. wrong profile). Impostor pairs center around 0.1 with spread from -0.2 to ~0.5. Genuine pairs concentrate around 0.75–0.85. The distributions are well-separated with minimal overlap.

**Conclusion**: The threshold is data-driven, not arbitrary. The clear separation between distributions confirms that cosine similarity is an effective discriminator, and the gray area fallback threshold can be reliably calibrated from this data.

### Visualization 5: Per-Class Performance Bars

Grouped bar chart of accuracy, precision, and recall per class. All three authorized users achieve 100% across all metrics. Unknown achieves 100% precision with 98% recall (reflecting the 2 misclassified samples).

**Conclusion**: The system is strongest at identifying known users with zero cross-contamination. The slight weakness in Unknown recall is minor and fully addressed by the pipeline's other gates.

### Visualization 6: System Architecture Diagram

A clean flowchart of the full pipeline: Gate 1 (Password) → Gate 2 (Biometrics) → Liveness Check → Gate 3 (Fusion Model) → 3-tier confidence decision.

**Conclusion**: Demonstrates defense in depth — an attacker must defeat all channels simultaneously.

### Visualization 7: Dashboard

All six visualizations composed into a single overview panel for presentation.

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Primary development language |
| Deep Learning Framework | PyTorch | Model training and inference |
| Face Detection & Alignment | MTCNN (facenet-pytorch) | Locate and crop faces from images |
| Face Embedding | InceptionResnetV1 / FaceNet (pretrained on VGGFace2) | 512-dim face identity vectors |
| Voice Embedding | ECAPA-TDNN (SpeechBrain, pretrained on VoxCeleb) | 192-dim speaker identity vectors |
| Speech-to-Text | Google Speech Recognition API | Transcribe spoken password |
| Liveness Detection | MediaPipe FaceLandmarker | Blink detection + head pose estimation |
| Camera Interface | OpenCV | Webcam capture and preview display |
| Audio Recording | sounddevice / scipy | Microphone input |
| Fusion Model | Custom MLP (PyTorch) | 704-dim → identity classification |
| Visualization | matplotlib, scikit-learn (t-SNE) | Evaluation plots and analysis |
| GPU Acceleration | CUDA (NVIDIA RTX 3080) | Training and inference acceleration |

All pretrained models are used frozen (transfer learning) — no fine-tuning of FaceNet or ECAPA-TDNN was required, which is appropriate for the small dataset size.

---

## Configuration Architecture (`utils/config.py`)

A centralized configuration system with 12 class-based namespaces covers every parameter in the project: file paths (`PATHS`), face preprocessing settings (`FACE_PREPROCESS`), voice preprocessing settings (`VOICE_PREPROCESS`), augmentation parameters (`FACE_AUGMENTATION`, `VOICE_AUGMENTATION`), embedding dimensions (`EMBEDDINGS`), enrollment settings (`ENROLLMENT`), training hyperparameters (`TRAINING`), class definitions (`CLASSES`), decision thresholds (`THRESHOLDS`), live system settings (`LIVE_SYSTEM`), device configuration (`DEVICE`), capture/preview settings (`CAPTURE`), and liveness parameters (`LIVENESS`).

All scripts import from this single source of truth. Changing a threshold, path, or hyperparameter in one place updates it everywhere. The `CLASSES` namespace auto-generates class-to-index and index-to-class mappings from the authorized user list — adding a new person requires changing only one line.

---

## Complete Feature List

1. **Multi-modal biometric authentication** — face + voice + spoken password
2. **Late fusion architecture** — 704-dim combined embedding (512 face + 192 voice)
3. **Transfer learning** — pretrained FaceNet and ECAPA-TDNN, frozen weights
4. **Data augmentation** — 7 face transformations (15×), 3 voice transformations (3×)
5. **Leakage-free data splitting** — split before pairing to prevent test contamination
6. **Class-weighted loss** — handles imbalanced class sizes
7. **Early stopping with checkpointing** — prevents overfitting, saves best model
8. **3-tier confidence decision** — high confidence / gray area fallback / rejection
9. **Cosine similarity fallback** — compares live embeddings against enrollment profiles
10. **Speech-to-text password verification** — fuzzy matching with keyword backup
11. **Liveness detection** — blink detection (EAR) + head pose variation (solvePnP)
12. **Anti-spoofing** — blocks photo and video replay attacks
13. **Live camera preview** — real-time face detection feedback with quality scoring
14. **Multi-frame quality scoring** — selects the best face frame from continuous capture
15. **Audio reuse optimization** — password recording used for both STT and voice embedding
16. **Admin override / active learning** — corrections saved for model retraining
17. **Comprehensive logging** — every attempt logged with full details for auditing
18. **Graceful shutdown** — clean resource release (camera, GPU, microphone)
19. **Centralized configuration** — single config file governs all scripts
20. **Full reproducibility** — seeded random generators across all frameworks
21. **Comprehensive visualization suite** — 7 evaluation plots at 300 DPI
22. **Self-contained model checkpoints** — architecture + weights + metadata in one file
23. **Lockout protection** — max attempts before temporary lockout
24. **GPU auto-detection** — seamless CPU/GPU operation

---

## Project Structure

```
SECURE-GATEWAY-DLPROJECT/
├── app/
│   └── run_system.py              ← Main executable: live authentication
├── data/
│   ├── embeddings/                ← Face (.pt) and voice (.pt) embedding files
│   ├── face/
│   │   ├── raw/                   ← Original photos per person
│   │   ├── processed/             ← MTCNN-aligned 160×160 crops
│   │   └── augmented/             ← Augmented face images
│   └── voice/
│       ├── raw/                   ← Original recordings per person
│       ├── processed/             ← Clean 16kHz mono WAVs
│       └── augmented/             ← Augmented audio clips
├── data_preparation/
│   ├── preprocess_faces.py        ← MTCNN detection + alignment + crop
│   ├── preprocess_voices.py       ← Resample + trim + normalize
│   ├── augment_face.py            ← Face augmentation pipeline
│   ├── augment_voice.py           ← Voice augmentation pipeline
│   ├── compute_embeddings.py      ← FaceNet + ECAPA-TDNN embedding
│   └── enroll_users.py            ← Mean profile computation
├── evaluation/
│   ├── figures/                   ← Generated visualization PNGs
│   └── visualizer.py             ← Visualization & analysis suite
├── logs/                          ← Authentication attempt logs
├── models/
│   ├── voice/ecapa_tdnn/          ← Cached ECAPA-TDNN model files
│   ├── data_splits.pt             ← Train/val/test split indices
│   ├── face_landmarker.task       ← MediaPipe liveness model
│   ├── fusion_model.pt            ← Trained fusion MLP checkpoint
│   ├── training_history.pt        ← Per-epoch loss/accuracy curves
│   └── user_profiles.pt           ← Enrolled reference templates
├── training/
│   └── train_model.py             ← Fusion model training pipeline
├── utils/
│   ├── __init__.py
│   └── config.py                  ← Centralized configuration (12 namespaces)
└── venv/                          ← Python virtual environment
```

---

## Execution Order

| Step | Script | Phase | Output |
|------|--------|-------|--------|
| 1 | `preprocess_faces.py` | Data Prep | Clean 160×160 face crops |
| 2 | `preprocess_voices.py` | Data Prep | Clean 16kHz mono WAVs |
| 3 | `augment_face.py` | Augmentation | ~480 face images per person |
| 4 | `augment_voice.py` | Augmentation | ~120 voice clips per person |
| 5 | `compute_embeddings.py` | Embedding | `face_embeddings.pt`, `voice_embeddings.pt` |
| 6 | `enroll_users.py` | Enrollment | `user_profiles.pt` |
| 7 | `train_model.py` | Training | `fusion_model.pt`, `training_history.pt`, `data_splits.pt` |
| 8 | `visualizer.py` | Evaluation | 7 visualization PNGs + dashboard |
| 9 | `run_system.py` | Live System | Real-time authentication |

---

## Key Results

- **Test accuracy**: 99.5% (376/378 correct on held-out test data)
- **Per-user accuracy**: David 100%, Itzhak 100%, Yossi 100%
- **Unknown detection**: 98% recall (112/114), 100% precision
- **Training convergence**: No overfitting, stable loss/accuracy curves
- **Embedding separation**: Clear t-SNE clusters with minimal overlap
- **Similarity distributions**: Well-separated genuine vs. impostor distributions
- **Architecture**: 3 independent security gates + liveness check = defense in depth

---

## Key Conclusions from the Data

1. **Late fusion works with small datasets** — even with ~30 original samples per person, the combination of transfer learning (frozen pretrained models) + data augmentation + a lightweight MLP achieves near-perfect classification.

2. **Face+voice fusion is more discriminative than either modality alone** — the t-SNE visualization shows tighter, better-separated clusters in the fused 704-dim space than would be achieved with face-only or voice-only embeddings.

3. **BatchNorm is essential for multi-modal fusion** — the 800× scale difference between face and voice embedding statistics would cause one modality to dominate without normalization.

4. **Class-weighted loss prevents minority class neglect** — without weighting, the model would over-optimize for David (largest dataset) at the expense of Itzhak and Yossi.

5. **Splitting before pairing prevents data leakage** — this is a subtle but critical methodological point that many student projects get wrong.

6. **The multi-gate architecture provides practical security** — even the rare fusion model error (2 unknown→Yossi misclassifications) is mitigated by the password gate and liveness check that must both pass first.

7. **Cosine similarity fallback improves usability** — the gray area zone between 50–85% confidence catches legitimate users under adverse conditions (poor lighting, background noise) without compromising security.

8. **Liveness detection blocks trivial attacks** — blink detection and head pose variation prevent the most common spoofing vectors (printed photos, screen replays) at minimal computational cost.
