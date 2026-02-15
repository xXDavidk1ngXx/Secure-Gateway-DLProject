"""
Voice Data Preprocessing Pipeline
===================================
Takes raw audio recordings from data/voice/raw/<person_name>/
Resamples, trims silence, normalizes volume, and validates audio quality.
Saves clean audio files to data/voice/processed/<person_name>/

Expected input structure:
    data/voice/raw/
    ├── david/
    │   ├── password/       ← 10 recordings of "my voice is my password"
    │   ├── short/          ← 10 short phrases
    │   └── long/           ← 10 longer sentences
    ├── itzhak/
    │   ├── password/
    │   ├── short/
    │   └── long/
    └── yossi/
        ├── password/
        ├── short/
        └── long/

Output structure (mirrors input):
    data/voice/processed/
    ├── david/
    │   ├── password/       ← Clean 16kHz mono WAV files
    │   ├── short/
    │   └── long/
    └── ...

Usage:
    python data_preparation/preprocess_voices.py

Requirements:
    pip install torchaudio torch soundfile tqdm numpy
    (soundfile needs libsndfile: apt install libsndfile1 on Ubuntu)
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


# ============================================================
#  CONFIGURATION — adjust these to fit your setup
# ============================================================

# Paths (relative to project root)
RAW_DATA_DIR = "data/voice/raw"                # Input: raw recordings organized by person/type
PROCESSED_DATA_DIR = "data/voice/processed"    # Output: clean audio files

# Audio processing settings
TARGET_SAMPLE_RATE = 16000   # 16kHz — standard for speech models (SpeechBrain, Whisper, etc.)
                              # Most pretrained speaker verification models expect 16kHz
                              # DO NOT change this unless your model specifically requires something else

TARGET_CHANNELS = 1           # Mono — speech models expect single-channel audio
                              # If your recordings are stereo, they'll be averaged to mono

# Silence trimming settings
TRIM_SILENCE = True           # Whether to trim leading/trailing silence
SILENCE_THRESHOLD_DB = -40    # Audio below this dB level is considered "silence"
                              # -40 dB is a good default for most recording environments
                              # If you're in a noisy room, try -30 dB
                              # If your mic is very clean, try -50 dB

MIN_SILENCE_DURATION = 0.1    # Minimum silence duration (seconds) to trigger trimming
                              # Prevents trimming brief pauses within speech

SILENCE_PADDING = 0.15        # Seconds of silence to keep at start/end after trimming
                              # A small pad prevents clipping the first/last phoneme
                              # 0.15s is enough context without wasting space

# Volume normalization
NORMALIZE_VOLUME = True       # Whether to normalize audio amplitude
PEAK_NORMALIZE_DB = -1.0      # Target peak amplitude in dB
                              # -1.0 dB leaves a tiny bit of headroom to avoid clipping
                              # This ensures all recordings have consistent volume levels

# Quality checks
MIN_DURATION_SEC = 0.5        # Minimum valid recording duration (seconds)
                              # Anything shorter is probably a mistake / empty file
MAX_DURATION_SEC = 30.0       # Maximum valid recording duration (seconds)
                              # Catches accidentally long recordings
MIN_RMS_DB = -50              # Minimum RMS energy — filters out silent/empty recordings
                              # If a "recording" is just ambient noise, it'll be below -50 dB

# Recording categories (subfolder names)
RECORDING_CATEGORIES = ["password", "short", "long"]


# ============================================================
#  STEP 1: Load and validate a single audio file
# ============================================================

def load_audio(file_path):
    """
    Load an audio file and return the waveform tensor + sample rate.

    torchaudio supports: WAV, MP3, FLAC, OGG, M4A, and more.
    The waveform is returned as a tensor of shape (channels, num_samples).

    Returns:
        waveform (torch.Tensor): Audio waveform, shape (channels, samples)
        sample_rate (int): Original sample rate of the file
        OR
        None, None if loading fails
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except Exception as e:
        print(f"  [ERROR] Could not load audio: {file_path} — {e}")
        return None, None


def validate_audio(waveform, sample_rate, filename):
    """
    Run quality checks on a loaded audio file.

    Checks:
      1. Duration is within acceptable range
      2. Audio has enough energy (not silence)
      3. No NaN or Inf values (corrupted data)

    Returns:
        (is_valid, reason): Tuple of (bool, str)
    """
    duration = waveform.shape[1] / sample_rate

    # Check duration
    if duration < MIN_DURATION_SEC:
        return False, f"too short ({duration:.2f}s < {MIN_DURATION_SEC}s)"
    if duration > MAX_DURATION_SEC:
        return False, f"too long ({duration:.2f}s > {MAX_DURATION_SEC}s)"

    # Check for corrupted data
    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        return False, "contains NaN or Inf values (corrupted)"

    # Check RMS energy (is there actual audio content?)
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms == 0:
        return False, "completely silent (zero RMS)"

    rms_db = 20 * torch.log10(rms + 1e-10)  # Convert to dB
    if rms_db < MIN_RMS_DB:
        return False, f"too quiet (RMS={rms_db:.1f} dB < {MIN_RMS_DB} dB)"

    return True, "ok"


# ============================================================
#  STEP 2: Audio preprocessing functions
# ============================================================

def convert_to_mono(waveform):
    """
    Convert multi-channel audio to mono by averaging channels.

    Input:  (channels, samples) — e.g., (2, 48000) for stereo
    Output: (1, samples)        — e.g., (1, 48000) mono

    Why mono?
      Speaker verification models process single-channel audio.
      Stereo just doubles the data with no benefit for voice tasks.
    """
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def resample_audio(waveform, orig_sr, target_sr):
    """
    Resample audio to the target sample rate.

    Why resample?
      Different recording devices use different sample rates (44.1kHz, 48kHz, etc.)
      Speech models expect a consistent rate — typically 16kHz.
      16kHz captures all speech-relevant frequencies (up to 8kHz by Nyquist).

    torchaudio.transforms.Resample uses polyphase filtering for high-quality resampling.
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    )
    return resampler(waveform)


def trim_silence_from_audio(waveform, sample_rate):
    """
    Remove leading and trailing silence from the audio.

    Algorithm:
      1. Compute energy (amplitude) of the waveform
      2. Find the first and last points where energy exceeds the threshold
      3. Trim everything outside those points
      4. Add a small padding back so we don't clip the speech onset

    This is important because:
      - Recordings often start/end with 1-2 seconds of silence
      - Silence adds no useful information for speaker verification
      - Trimming makes recordings more consistent in length
      - Reduces wasted computation during embedding extraction
    """
    if not TRIM_SILENCE:
        return waveform

    # Convert threshold from dB to linear amplitude
    threshold_linear = 10 ** (SILENCE_THRESHOLD_DB / 20)

    # Get absolute amplitude
    audio_abs = torch.abs(waveform[0])  # Work with the mono channel

    # Find where audio exceeds threshold
    above_threshold = audio_abs > threshold_linear

    if not above_threshold.any():
        # Entire recording is below threshold — return as-is (will be caught by validation)
        return waveform

    # Find first and last non-silent sample
    nonzero_indices = torch.where(above_threshold)[0]
    start_idx = nonzero_indices[0].item()
    end_idx = nonzero_indices[-1].item()

    # Add padding (convert seconds to samples)
    pad_samples = int(SILENCE_PADDING * sample_rate)
    start_idx = max(0, start_idx - pad_samples)
    end_idx = min(waveform.shape[1], end_idx + pad_samples)

    return waveform[:, start_idx:end_idx]


def normalize_volume(waveform):
    """
    Peak-normalize the audio to a target dB level.

    Why normalize?
      Different recordings have different volume levels depending on:
      - Distance from the microphone
      - Mic sensitivity settings
      - Speaking volume

      Normalization ensures all recordings have consistent amplitude,
      which helps the speaker embedding model treat them equally.

    We use peak normalization (scales based on the loudest sample)
    rather than RMS normalization, because it's simpler and sufficient
    when followed by model-level normalization.
    """
    if not NORMALIZE_VOLUME:
        return waveform

    # Find current peak amplitude
    peak = torch.max(torch.abs(waveform))
    if peak == 0:
        return waveform  # Avoid division by zero

    # Calculate target peak in linear scale
    target_peak = 10 ** (PEAK_NORMALIZE_DB / 20)

    # Scale the waveform
    gain = target_peak / peak
    return waveform * gain


# ============================================================
#  STEP 3: Process a single audio file (full pipeline)
# ============================================================

def process_single_audio(file_path):
    """
    Full preprocessing pipeline for one audio file:
      1. Load the audio
      2. Validate (duration, energy, corruption checks)
      3. Convert to mono
      4. Resample to 16kHz
      5. Trim silence
      6. Normalize volume
      7. Re-validate after processing

    Returns:
        processed_waveform (torch.Tensor): Clean audio (1, samples) at 16kHz
        metadata (dict): Info about the processing (duration, original SR, etc.)
        OR
        None, error_dict if processing fails
    """
    # Step 1: Load
    waveform, sample_rate = load_audio(file_path)
    if waveform is None:
        return None, {"error": "failed to load"}

    filename = os.path.basename(file_path)
    original_duration = waveform.shape[1] / sample_rate
    original_sr = sample_rate

    # Step 2: Initial validation
    is_valid, reason = validate_audio(waveform, sample_rate, filename)
    if not is_valid:
        return None, {"error": reason}

    # Step 3: Convert to mono
    waveform = convert_to_mono(waveform)

    # Step 4: Resample to target sample rate
    waveform = resample_audio(waveform, sample_rate, TARGET_SAMPLE_RATE)
    sample_rate = TARGET_SAMPLE_RATE

    # Step 5: Trim silence
    waveform = trim_silence_from_audio(waveform, sample_rate)

    # Step 6: Normalize volume
    waveform = normalize_volume(waveform)

    # Step 7: Post-processing validation
    final_duration = waveform.shape[1] / sample_rate
    if final_duration < MIN_DURATION_SEC:
        return None, {"error": f"too short after trimming ({final_duration:.2f}s)"}

    metadata = {
        "original_sample_rate": original_sr,
        "original_duration": round(original_duration, 2),
        "processed_duration": round(final_duration, 2),
        "final_sample_rate": sample_rate,
        "trimmed_seconds": round(original_duration - final_duration, 2),
    }

    return waveform, metadata


# ============================================================
#  STEP 4: Process all recordings for a single person
# ============================================================

def process_person(person_name, raw_dir, output_dir):
    """
    Process all audio files for one person across all categories
    (password, short, long).

    Reads from:  raw_dir/<person_name>/<category>/
    Saves to:    output_dir/<person_name>/<category>/
    """
    person_raw_dir = os.path.join(raw_dir, person_name)
    person_out_dir = os.path.join(output_dir, person_name)

    # Supported audio formats
    valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".webm", ".opus"}

    overall_stats = {
        "total": 0, "success": 0, "failed": 0,
        "total_raw_duration": 0, "total_processed_duration": 0,
    }
    all_failed = []

    # Check if person folder has category subfolders or flat structure
    has_categories = any(
        os.path.isdir(os.path.join(person_raw_dir, cat))
        for cat in RECORDING_CATEGORIES
    )

    if has_categories:
        categories_to_process = RECORDING_CATEGORIES
    else:
        # Flat structure — all files directly in person folder
        categories_to_process = ["."]
        print(f"  [INFO] No category subfolders found for '{person_name}', processing flat structure")

    for category in categories_to_process:
        if category == ".":
            input_folder = person_raw_dir
            output_folder = person_out_dir
            display_name = person_name
        else:
            input_folder = os.path.join(person_raw_dir, category)
            output_folder = os.path.join(person_out_dir, category)
            display_name = f"{person_name}/{category}"

        if not os.path.exists(input_folder):
            print(f"  [WARNING] Category folder not found: {input_folder}")
            continue

        os.makedirs(output_folder, exist_ok=True)

        # Get all audio files
        audio_files = [
            f for f in sorted(os.listdir(input_folder))
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if not audio_files:
            print(f"  [WARNING] No audio files in {input_folder}")
            continue

        print(f"\n  Processing {len(audio_files)} files for '{display_name}'...")

        category_stats = {"total": len(audio_files), "success": 0, "failed": 0}
        failed_files = []

        for filename in tqdm(audio_files, desc=f"  {display_name}", leave=True):
            file_path = os.path.join(input_folder, filename)

            # Process the audio file
            processed, result = process_single_audio(file_path)

            if processed is None:
                category_stats["failed"] += 1
                failed_files.append((filename, result.get("error", "unknown")))
                continue

            # Save as WAV (consistent format, lossless, widely supported)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_folder, output_filename)

            torchaudio.save(
                output_path,
                processed,
                TARGET_SAMPLE_RATE,
                encoding="PCM_S",       # 16-bit signed PCM
                bits_per_sample=16,     # Standard for speech processing
            )

            category_stats["success"] += 1
            overall_stats["total_raw_duration"] += result["original_duration"]
            overall_stats["total_processed_duration"] += result["processed_duration"]

        # Update overall stats
        overall_stats["total"] += category_stats["total"]
        overall_stats["success"] += category_stats["success"]
        overall_stats["failed"] += category_stats["failed"]
        all_failed.extend([(f"{category}/{f}", r) for f, r in failed_files])

        print(f"  {display_name}: {category_stats['success']}/{category_stats['total']} successful")

    # Report results for this person
    print(f"\n  Results for '{person_name}':")
    print(f"    ✓ Success:  {overall_stats['success']}/{overall_stats['total']}")
    print(f"    ✗ Failed:   {overall_stats['failed']}")
    print(f"    ⏱ Duration: {overall_stats['total_raw_duration']:.1f}s raw → {overall_stats['total_processed_duration']:.1f}s processed")

    if all_failed:
        print(f"  Failed files:")
        for fname, reason in all_failed:
            print(f"    - {fname}: {reason}")

    return overall_stats


# ============================================================
#  STEP 5: Run the full pipeline
# ============================================================

def run_preprocessing():
    """
    Main function — processes all people in the raw data directory.
    """
    print("=" * 60)
    print("  VOICE DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Verify input directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"\n[ERROR] Raw data directory not found: {RAW_DATA_DIR}")
        print(f"Expected structure:")
        print(f"  {RAW_DATA_DIR}/")
        print(f"    ├── david/")
        print(f"    │   ├── password/    ← 10 recordings of 'my voice is my password'")
        print(f"    │   ├── short/       ← 10 short phrases")
        print(f"    │   └── long/        ← 10 longer sentences")
        print(f"    ├── itzhak/")
        print(f"    └── yossi/")
        sys.exit(1)

    # Find all person folders
    person_folders = [
        d for d in sorted(os.listdir(RAW_DATA_DIR))
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
        and not d.startswith(".")
    ]

    if not person_folders:
        print(f"\n[ERROR] No person folders found in {RAW_DATA_DIR}")
        sys.exit(1)

    print(f"\nFound {len(person_folders)} people: {', '.join(person_folders)}")
    print(f"Input:  {RAW_DATA_DIR}")
    print(f"Output: {PROCESSED_DATA_DIR}")
    print(f"Settings:")
    print(f"  Sample rate:     {TARGET_SAMPLE_RATE} Hz")
    print(f"  Channels:        {'Mono' if TARGET_CHANNELS == 1 else 'Stereo'}")
    print(f"  Trim silence:    {TRIM_SILENCE} (threshold={SILENCE_THRESHOLD_DB} dB)")
    print(f"  Normalize:       {NORMALIZE_VOLUME} (peak={PEAK_NORMALIZE_DB} dB)")
    print(f"  Duration range:  {MIN_DURATION_SEC}s – {MAX_DURATION_SEC}s")

    # Create output directory
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Process each person
    all_stats = {}
    for person_name in person_folders:
        print(f"\n{'─' * 50}")
        print(f"  Person: {person_name}")
        print(f"{'─' * 50}")
        stats = process_person(person_name, RAW_DATA_DIR, PROCESSED_DATA_DIR)
        all_stats[person_name] = stats

    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE — SUMMARY")
    print("=" * 60)

    total_files = sum(s["total"] for s in all_stats.values())
    total_success = sum(s["success"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    total_raw_dur = sum(s["total_raw_duration"] for s in all_stats.values())
    total_proc_dur = sum(s["total_processed_duration"] for s in all_stats.values())

    print(f"\n  {'Person':<15} {'Total':<8} {'Success':<10} {'Failed':<10} {'Raw Dur':<12} {'Clean Dur':<12}")
    print(f"  {'-' * 67}")
    for person, stats in all_stats.items():
        raw_d = f"{stats['total_raw_duration']:.1f}s"
        proc_d = f"{stats['total_processed_duration']:.1f}s"
        print(f"  {person:<15} {stats['total']:<8} {stats['success']:<10} {stats['failed']:<10} {raw_d:<12} {proc_d:<12}")
    print(f"  {'-' * 67}")
    raw_t = f"{total_raw_dur:.1f}s"
    proc_t = f"{total_proc_dur:.1f}s"
    print(f"  {'TOTAL':<15} {total_files:<8} {total_success:<10} {total_failed:<10} {raw_t:<12} {proc_t:<12}")

    success_rate = (total_success / total_files * 100) if total_files > 0 else 0
    print(f"\n  Overall success rate: {success_rate:.1f}%")

    if total_raw_dur > 0:
        trim_pct = (1 - total_proc_dur / total_raw_dur) * 100
        print(f"  Silence trimmed:     {trim_pct:.1f}% of total duration removed")

    if total_failed > 0:
        print(f"\n  Tips for failed recordings:")
        print(f"  - Make sure recordings have audible speech (not just silence)")
        print(f"  - Check that files aren't corrupted (try playing them manually)")
        print(f"  - If too many fail from silence trimming, lower SILENCE_THRESHOLD_DB to -50")
        print(f"  - For very short recordings, lower MIN_DURATION_SEC")

    print(f"\n  Processed audio saved to: {PROCESSED_DATA_DIR}/")
    print(f"  You can now proceed to voice data augmentation.\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    run_preprocessing()