"""
Voice Data Augmentation Pipeline
==================================
Takes clean audio from data/voice/processed/<person_name>/
Generates N augmented versions to simulate real-world recording variation.
Saves augmented audio to data/voice/augmented/<person_name>/

Why augment voice data?
    Your security system will encounter people speaking in different rooms,
    at different distances from the mic, with background noise, with slight
    vocal changes (tired, sick, excited). Augmentation teaches the embedding
    model to handle all of this by creating training examples that mimic
    these conditions.

    With ~30 clips per person × 3 augmentations = ~120 clips per person.
    We use fewer augmentations than faces because:
      1. ECAPA-TDNN (the voice embedding model) is already quite robust
      2. Audio augmentation is more sensitive — heavy distortion destroys identity
      3. Each clip already contains thousands of audio frames

Expected input:
    data/voice/processed/
    ├── david/
    │   ├── password/          ← clean 16kHz WAV files
    │   │   ├── rec_01.wav
    │   │   └── ...
    │   ├── short/
    │   └── long/
    ├── itzhak/
    └── yossi/

Output:
    data/voice/augmented/
    ├── david/
    │   ├── password/
    │   │   ├── rec_01_original.wav      ← copy of the clean original
    │   │   ├── rec_01_aug_01.wav        ← augmented version 1
    │   │   ├── rec_01_aug_02.wav        ← augmented version 2
    │   │   ├── rec_01_aug_03.wav        ← augmented version 3
    │   │   └── ...
    │   ├── short/
    │   └── long/
    ├── itzhak/
    └── yossi/

Usage:
    python data_preparation/augment_voices.py

Requirements:
    pip install torch torchaudio numpy tqdm soundfile
"""

import os
import sys
import random
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


# ============================================================
#  CONFIGURATION — adjust these to fit your setup
# ============================================================

# Paths (relative to project root)
PROCESSED_DATA_DIR = "data/voice/processed"   # Input: clean preprocessed audio
AUGMENTED_DATA_DIR = "data/voice/augmented"   # Output: original + augmented audio

# Augmentation settings
AUGMENTATIONS_PER_CLIP = 3      # Number of augmented versions per original clip
                                 # With 30 originals: 30 + (30 × 3) = 120 clips per person
                                 # We keep this lower than faces because:
                                 #   - Voice models are more robust to variation
                                 #   - Heavy audio augmentation can destroy identity cues
                                 #   - Each clip already has many audio frames

COPY_ORIGINALS = True            # Copy original clean audio to augmented folder
                                 # Makes the augmented folder self-contained

TARGET_SAMPLE_RATE = 16000       # Must match preprocessing output

RANDOM_SEED = 42                 # For reproducibility (None for random)


# ============================================================
#  STEP 1: Define individual augmentation functions
# ============================================================
#
#  Each augmentation simulates a real-world condition:
#
#  ┌─────────────────────────┬────────────────────────────────────────┐
#  │ Augmentation            │ Real-world scenario                    │
#  ├─────────────────────────┼────────────────────────────────────────┤
#  │ Add Gaussian noise      │ Background noise (office, street, fan) │
#  │ Pitch shift ±2 semi     │ Vocal variation (morning vs evening)   │
#  │ Time stretch ±10%       │ Speaking speed variation (rushed/calm) │
#  │ Volume perturbation     │ Mic distance variation                 │
#  │ Simple reverb (echo)    │ Room acoustics (hallway, small room)   │
#  └─────────────────────────┴────────────────────────────────────────┘
#
#  IMPORTANT — what we do NOT apply:
#  - Heavy pitch shifts (>3 semitones would change perceived identity)
#  - Large time stretches (>15% creates unnatural artifacts)
#  - Frequency masking (destroys formant structure that carries identity)
#  - Codec compression (inconsistent and hard to control)


def add_gaussian_noise(waveform, snr_range=(15, 30)):
    """
    Add random Gaussian noise at a random SNR (Signal-to-Noise Ratio).
    
    SNR in dB controls how loud the noise is relative to the speech:
      30 dB = very faint background hiss (quiet room)
      20 dB = noticeable background noise (office with AC)
      15 dB = significant noise (busy café)
      10 dB = very noisy (would be hard to hear — too aggressive for our use)
    
    We use 15–30 dB range: realistic but not destructive.
    
    Args:
        waveform: Tensor of shape (1, num_samples)
        snr_range: Tuple of (min_snr_db, max_snr_db)
    
    Returns:
        Noisy waveform tensor
    """
    snr_db = random.uniform(*snr_range)
    
    # Calculate signal power
    signal_power = torch.mean(waveform ** 2)
    
    # Calculate required noise power from SNR formula:
    #   SNR(dB) = 10 * log10(signal_power / noise_power)
    #   noise_power = signal_power / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise with the calculated power
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    
    return waveform + noise


def pitch_shift(waveform, sample_rate, semitones_range=(-2, 2)):
    """
    Shift the pitch of the audio by a random number of semitones.
    
    Why pitch shift?
      A person's voice pitch varies naturally — they might sound slightly
      higher when excited or slightly lower when tired. A ±2 semitone
      range covers this natural variation without changing perceived identity.
    
    Method: Resample-based pitch shifting
      1. Time-stretch the audio (changes speed + pitch)
      2. Resample to restore original duration (keeps pitch change, fixes speed)
    
    This is simpler than phase-vocoder methods but works well for small shifts.
    
    Args:
        waveform: Tensor of shape (1, num_samples)
        sample_rate: Audio sample rate (16000)
        semitones_range: Tuple of (min, max) semitones to shift
    
    Returns:
        Pitch-shifted waveform tensor
    """
    semitones = random.uniform(*semitones_range)
    
    # Pitch shift factor: 2^(semitones/12)
    # +2 semitones → factor ≈ 1.122 (higher pitch)
    # -2 semitones → factor ≈ 0.891 (lower pitch)
    factor = 2 ** (semitones / 12.0)
    
    # Resample: changing the "assumed" sample rate effectively changes pitch
    # If we tell the resampler the audio was recorded at a higher rate,
    # it will downsample → lower pitch (and vice versa)
    resampler = torchaudio.transforms.Resample(
        orig_freq=int(sample_rate * factor),
        new_freq=sample_rate,
    )
    shifted = resampler(waveform)
    
    # The resampling changes the length — trim or pad to match original
    original_length = waveform.shape[1]
    if shifted.shape[1] > original_length:
        shifted = shifted[:, :original_length]
    elif shifted.shape[1] < original_length:
        padding = original_length - shifted.shape[1]
        shifted = torch.nn.functional.pad(shifted, (0, padding))
    
    return shifted


def time_stretch(waveform, rate_range=(0.9, 1.1)):
    """
    Change the speed of the audio without changing pitch.
    
    Why time stretch?
      People speak at different speeds — rushed when nervous, slower when
      thinking. A ±10% variation is realistic and subtle.
    
    Simple method: Linear interpolation resampling.
    This does slightly affect pitch (unlike a proper phase vocoder),
    but at ±10% the effect is negligible and the identity is preserved.
    
    Args:
        waveform: Tensor of shape (1, num_samples)
        rate_range: Tuple of (min_rate, max_rate)
                    >1.0 = faster (shorter audio)
                    <1.0 = slower (longer audio)
    
    Returns:
        Time-stretched waveform tensor (same length as input)
    """
    rate = random.uniform(*rate_range)
    
    # Use torchaudio's resample as a simple stretching mechanism
    original_length = waveform.shape[1]
    
    # Stretch by resampling
    stretched = torchaudio.functional.resample(
        waveform,
        orig_freq=int(16000 * rate),
        new_freq=16000,
    )
    
    # Trim or pad to original length
    if stretched.shape[1] > original_length:
        stretched = stretched[:, :original_length]
    elif stretched.shape[1] < original_length:
        padding = original_length - stretched.shape[1]
        stretched = torch.nn.functional.pad(stretched, (0, padding))
    
    return stretched


def volume_perturbation(waveform, db_range=(-3, 3)):
    """
    Randomly change the volume (gain) of the audio.
    
    Why volume perturbation?
      When someone approaches the mic, they might be closer or further away.
      Volume changes by ±3 dB are subtle but realistic.
      The preprocessing normalized everything to a consistent level —
      this adds back some natural variation.
    
    Args:
        waveform: Tensor of shape (1, num_samples)
        db_range: Tuple of (min_db, max_db) gain to apply
    
    Returns:
        Volume-adjusted waveform tensor
    """
    gain_db = random.uniform(*db_range)
    
    # Convert dB to linear gain: gain = 10^(dB/20)
    gain = 10 ** (gain_db / 20.0)
    
    result = waveform * gain
    
    # Clip to prevent digital clipping
    result = torch.clamp(result, -1.0, 1.0)
    
    return result


def add_simple_reverb(waveform, sample_rate, decay_range=(0.1, 0.4), delay_ms_range=(20, 50)):
    """
    Add a simple reverb effect by mixing delayed copies of the signal.
    
    Why reverb?
      Different rooms have different acoustic properties. A hallway has
      long reverb, a small office has short reverb. This simulates the
      person speaking in different spaces.
    
    Method: Simple feedback delay (not a full convolution reverb).
      We add a single delayed, attenuated copy of the signal.
      This is a crude approximation but effective for data augmentation.
    
    Args:
        waveform: Tensor of shape (1, num_samples)
        sample_rate: Audio sample rate
        decay_range: How loud the echo is (0 = none, 1 = full volume)
        delay_ms_range: Echo delay in milliseconds
    
    Returns:
        Reverb-added waveform tensor
    """
    decay = random.uniform(*decay_range)
    delay_ms = random.uniform(*delay_ms_range)
    
    # Convert delay from milliseconds to samples
    delay_samples = int(sample_rate * delay_ms / 1000.0)
    
    # Create the delayed version
    delayed = torch.zeros_like(waveform)
    if delay_samples < waveform.shape[1]:
        delayed[:, delay_samples:] = waveform[:, :-delay_samples] * decay
    
    result = waveform + delayed
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(result))
    if max_val > 1.0:
        result = result / max_val
    
    return result


# ============================================================
#  STEP 2: Apply a random combination of augmentations
# ============================================================

def augment_single_clip(waveform, sample_rate):
    """
    Apply a random combination of augmentations to one audio clip.
    
    Each augmentation is applied with a certain probability.
    This means each augmented version gets a DIFFERENT random
    combination, creating diverse training examples.
    
    The order matters slightly:
      1. Time/pitch modifications first (structural changes)
      2. Then noise/reverb (additive effects)
      3. Volume last (global scaling)
    
    Args:
        waveform: Tensor of shape (1, num_samples)
        sample_rate: Audio sample rate (16000)
    
    Returns:
        Augmented waveform tensor
    """
    augmented = waveform.clone()
    
    # 1. Pitch shift (40% chance)
    if random.random() < 0.4:
        augmented = pitch_shift(augmented, sample_rate, semitones_range=(-2, 2))
    
    # 2. Time stretch (40% chance)
    if random.random() < 0.4:
        augmented = time_stretch(augmented, rate_range=(0.9, 1.1))
    
    # 3. Add background noise (50% chance — most common real-world issue)
    if random.random() < 0.5:
        augmented = add_gaussian_noise(augmented, snr_range=(15, 30))
    
    # 4. Add reverb (30% chance)
    if random.random() < 0.3:
        augmented = add_simple_reverb(augmented, sample_rate)
    
    # 5. Volume perturbation (50% chance)
    if random.random() < 0.5:
        augmented = volume_perturbation(augmented, db_range=(-3, 3))
    
    return augmented


# ============================================================
#  STEP 3: Process all clips for a single person
# ============================================================

def process_person(person_name, input_dir, output_dir):
    """
    Generate augmented audio for one person.
    Handles both category-based (password/short/long) and flat folder structures.
    
    Returns:
        dict with counts of originals, augmented, and failures
    """
    person_in_dir = os.path.join(input_dir, person_name)
    person_out_dir = os.path.join(output_dir, person_name)
    
    valid_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    
    # Check if person folder has category subfolders
    categories = ["password", "short", "long"]
    has_categories = any(
        os.path.isdir(os.path.join(person_in_dir, cat))
        for cat in categories
    )
    
    if has_categories:
        folders_to_process = categories
    else:
        folders_to_process = ["."]  # Flat structure
        print(f"  [INFO] No category subfolders for '{person_name}', processing flat structure")
    
    overall_stats = {"originals": 0, "augmented": 0, "failed": 0}
    
    for category in folders_to_process:
        if category == ".":
            input_folder = person_in_dir
            output_folder = person_out_dir
            display_name = person_name
        else:
            input_folder = os.path.join(person_in_dir, category)
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
        
        print(f"\n  Augmenting {len(audio_files)} clips for '{display_name}' "
              f"(×{AUGMENTATIONS_PER_CLIP} each)...")
        
        for filename in tqdm(audio_files, desc=f"  {display_name}", leave=True):
            file_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                waveform, sr = torchaudio.load(file_path)
            except Exception as e:
                print(f"  [ERROR] Could not load {filename}: {e}")
                overall_stats["failed"] += 1
                continue
            
            # Resample if needed (should already be 16kHz from preprocessing)
            if sr != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # --- Save the original ---
            if COPY_ORIGINALS:
                original_path = os.path.join(output_folder, f"{base_name}_original.wav")
                torchaudio.save(original_path, waveform, TARGET_SAMPLE_RATE)
                overall_stats["originals"] += 1
            
            # --- Generate augmented versions ---
            for aug_idx in range(1, AUGMENTATIONS_PER_CLIP + 1):
                try:
                    augmented = augment_single_clip(waveform, TARGET_SAMPLE_RATE)
                    
                    aug_filename = f"{base_name}_aug_{aug_idx:02d}.wav"
                    aug_path = os.path.join(output_folder, aug_filename)
                    torchaudio.save(aug_path, augmented, TARGET_SAMPLE_RATE)
                    
                    overall_stats["augmented"] += 1
                    
                except Exception as e:
                    print(f"  [ERROR] Augmentation {aug_idx} failed for {filename}: {e}")
                    overall_stats["failed"] += 1
    
    total = overall_stats["originals"] + overall_stats["augmented"]
    print(f"\n  Results for '{person_name}':")
    print(f"    ✓ Originals copied: {overall_stats['originals']}")
    print(f"    ✓ Augmented:        {overall_stats['augmented']}")
    print(f"    ✓ Total clips:      {total}")
    print(f"    ✗ Failed:           {overall_stats['failed']}")
    
    return overall_stats


# ============================================================
#  STEP 4: Run the full augmentation pipeline
# ============================================================

def run_augmentation():
    """
    Main function — augments all people in the processed voice directory.
    """
    print("=" * 60)
    print("  VOICE DATA AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Set random seed
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        print(f"\n  Random seed: {RANDOM_SEED} (results are reproducible)")
    
    # Verify input directory
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"\n[ERROR] Processed data directory not found: {PROCESSED_DATA_DIR}")
        print(f"Run preprocess_voices.py first!")
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
    print(f"  Settings: {AUGMENTATIONS_PER_CLIP} augmentations per clip")
    
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
    
    # Process each person
    all_stats = {}
    for person_name in person_folders:
        print(f"\n{'─' * 50}")
        stats = process_person(person_name, PROCESSED_DATA_DIR, AUGMENTED_DATA_DIR)
        all_stats[person_name] = stats
    
    # ============================================================
    #  FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("  VOICE AUGMENTATION COMPLETE — SUMMARY")
    print("=" * 60)
    
    total_originals = sum(s["originals"] for s in all_stats.values())
    total_augmented = sum(s["augmented"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    total_clips = total_originals + total_augmented
    
    print(f"\n  {'Person':<15} {'Originals':<12} {'Augmented':<12} {'Total':<10} {'Failed':<10}")
    print(f"  {'-' * 59}")
    for person, stats in all_stats.items():
        total = stats["originals"] + stats["augmented"]
        print(f"  {person:<15} {stats['originals']:<12} {stats['augmented']:<12} {total:<10} {stats['failed']:<10}")
    print(f"  {'-' * 59}")
    print(f"  {'TOTAL':<15} {total_originals:<12} {total_augmented:<12} {total_clips:<10} {total_failed:<10}")
    
    print(f"\n  Augmented audio saved to: {AUGMENTED_DATA_DIR}/")
    print(f"\n  Next step: python data_preparation/compute_embeddings.py\n")


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    run_augmentation()