# metadata_csv_to_wavs.py
# Convert CSV with {filename}|{transcript} rows to WAV files using F5-TTS
# Optimized for large CSV files (10000+ rows) - model loaded once, efficient batch processing

# ========================
# 🔔 CONFIGURATION - CHANGE THESE VALUES
# ========================

# Input CSV file path (no header, format: {filename}|{transcript})
CSV_FILE = "/content/input.csv"

# Output directory for generated WAV files
OUTPUT_DIR = "./output"

# Reference audio and text for voice cloning
REF_AUDIO = "/content/drive/MyDrive/shared_rakion/voices/Đây là bằng hữu của ta TRIMMED.wav"
REF_TEXT = "Đây là bằng hữu của ta. Vị cô nương này không được khỏe. Các ngươi mau đưa nàng ấy vào trong chữa trị đi"

# Model configuration
MODEL_NAME = "F5TTS_Base"
VOCODER_NAME = "vocos"
CKPT_FILE = "/content/drive/MyDrive/shared_underwater/F5-TTS-Vietnamese-ViVoice_models/model_last.pt"
VOCAB_FILE = "/content/drive/MyDrive/shared_underwater/F5-TTS-Vietnamese-ViVoice_models/vocab.txt"

# Inference parameters
SPEED = 1.0
TARGET_RMS = 0.1
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0
REMOVE_SILENCE = False

# Resume mode: skip existing files (True = resume-safe)
RESUME_MODE = True

# ========================
# IMPORTS
# ========================

import os
import sys
import time
from pathlib import Path
from vinorm import TTSnorm

# Setup paths
os.chdir('/content')
sys.path.append("/content/drive/MyDrive/shared_underwater/F5_TTS_Vietnamese/src")

# Import F5-TTS core components directly
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    infer_process,
    load_model,
    load_vocoder,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT  # needed for model config
from omegaconf import OmegaConf
import soundfile as sf
from cached_path import cached_path


# ========================
# GLOBAL MODEL CACHE
# ========================
_model_cache = {
    'vocoder': None,
    'ema_model': None,
    'model_cfg': None,
    'vocoder_name': None,
    'vocab_file': None,
    'ckpt_file': None,
    'model_name': None,
}


def load_tts_model_once(
    model_name=MODEL_NAME,
    ckpt_file=CKPT_FILE,
    vocab_file=VOCAB_FILE,
    vocoder_name=VOCODER_NAME,
    load_vocoder_from_local=True
):
    """Load TTS model and vocoder ONCE and cache globally."""
    global _model_cache

    if (
        _model_cache['ema_model'] is not None and
        _model_cache['model_name'] == model_name and
        _model_cache['ckpt_file'] == ckpt_file and
        _model_cache['vocab_file'] == vocab_file and
        _model_cache['vocoder_name'] == vocoder_name
    ):
        print("✅ Using cached TTS model & vocoder")
        return _model_cache['ema_model'], _model_cache['vocoder']

    print(f"🧠 Loading TTS model: {model_name}...")

    # Load vocoder
    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    else:
        raise ValueError(f"Unsupported vocoder: {vocoder_name}")

    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=load_vocoder_from_local,
        local_path=vocoder_local_path
    )

    # Load model config
    model_cfg_path = f"configs/{model_name}.yaml"
    try:
        model_cfg = OmegaConf.load(model_cfg_path).model
    except Exception:
        # Fallback to package resources if local not found
        from importlib.resources import files
        pkg_cfg_path = files("f5_tts").joinpath(model_cfg_path)
        model_cfg = OmegaConf.load(str(pkg_cfg_path)).model

    model_cls = globals()[model_cfg.backbone]

    # Resolve checkpoint
    resolved_ckpt_file = ckpt_file
    if not ckpt_file:
        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
        if model_name == "F5TTS_Base":
            if vocoder_name == "vocos":
                ckpt_step = 1200000
            elif vocoder_name == "bigvgan":
                model_name = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model_name == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        resolved_ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))

    # Load model
    ema_model = load_model(
        model_cls,
        model_cfg.arch,
        resolved_ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file=vocab_file
    )

    # Cache globally
    _model_cache.update({
        'vocoder': vocoder,
        'ema_model': ema_model,
        'model_cfg': model_cfg,
        'vocoder_name': vocoder_name,
        'vocab_file': vocab_file,
        'ckpt_file': resolved_ckpt_file,
        'model_name': model_name,
    })

    print(f"✅ Model {model_name} loaded and cached.")
    return ema_model, vocoder


# ========================
# TEXT PROCESSING
# ========================
def post_process(text):
    """Clean up TTS output text."""
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace(" , , ", " , ")
    text = text.replace(" ,, ", " , ")
    text = text.replace('"', "")
    return " ".join(text.split())


# ========================
# INFERENCE WRAPPER (NO SUBPROCESS)
# ========================
def generate_speech_direct(
    ref_audio_proc,
    ref_text_proc,
    gen_text,
    speed=SPEED,
    vocoder_name=VOCODER_NAME,
    target_rms=TARGET_RMS,
    nfe_step=NFE_STEP,
    cfg_strength=CFG_STRENGTH,
    sway_sampling_coef=SWAY_SAMPLING_COEF,
    output_dir=OUTPUT_DIR,
    output_file="output.wav",
    remove_silence=REMOVE_SILENCE,
    vocab_file=VOCAB_FILE,
    ckpt_file=CKPT_FILE,
    model_name=MODEL_NAME,
):
    """
    Directly call infer_process without subprocess.
    Reuses globally loaded model and vocoder.
    """
    os.makedirs(output_dir, exist_ok=True)
    wave_path = Path(output_dir) / output_file

    # Load or reuse model
    ema_model, vocoder = load_tts_model_once(
        model_name=model_name,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        vocoder_name=vocoder_name
    )

    # Normalize generated text
    gen_text_clean = post_process(TTSnorm(gen_text))

    print(f"🚀 Generating: {output_file[:50]}...")

    try:
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio_proc,
            ref_text_proc,
            gen_text_clean,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=0.15,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=None,
        )

        # Save audio
        with open(wave_path, "wb") as f:
            sf.write(f.name, audio_segment, final_sample_rate)
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f"✅ Saved: {wave_path}")

        return str(wave_path)

    except Exception as e:
        print(f"❌ Error generating {output_file}: {e}")
        raise RuntimeError(f"F5-TTS inference failed: {e}") from e


# ========================
# CSV PARSING
# ========================
def parse_csv(csv_path):
    """
    Parse CSV file with no header.
    Each row format: {filename}|{transcript}
    Returns list of tuples: [(filename, transcript), ...]
    """
    entries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            # Split on first '|' only (transcript may contain '|')
            parts = line.split('|', 1)
            if len(parts) != 2:
                print(f"⚠️  Skipping line {line_num}: invalid format (expected filename|transcript)")
                continue
            
            filename, transcript = parts
            filename = filename.strip()
            transcript = transcript.strip()
            
            if not filename or not transcript:
                print(f"⚠️  Skipping line {line_num}: empty filename or transcript")
                continue
            
            # Ensure filename has .wav extension
            if not filename.lower().endswith('.wav'):
                filename = filename + '.wav'
            
            entries.append((filename, transcript))
    
    return entries


# ========================
# MAIN CSV → WAV PIPELINE
# ========================
def csv_to_wavs_optimized(
    csv_file=CSV_FILE,
    output_dir=OUTPUT_DIR,
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    preprocessed_voice=None,
    speed=SPEED,
    resume_mode=RESUME_MODE,
):
    """
    Convert CSV to WAVs with optimized workflow.
    Preprocess reference ONCE. Load model ONCE. Resume-safe.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse CSV
    entries = parse_csv(csv_file)
    total_rows = len(entries)
    print(f"\n📁 Parsed {total_rows} entries from {csv_file}")
    
    # Preprocess reference ONCE
    if preprocessed_voice is None:
        print("🎙️ Preprocessing reference voice...")
        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)
    else:
        print("⚡ Using cached reference voice")
        ref_audio_proc, ref_text_proc = preprocessed_voice

    generated_files = []
    skipped_count = 0
    error_count = 0

    for i, (filename, transcript) in enumerate(entries, start=1):
        output_path = os.path.join(output_dir, filename)

        # Resume: skip if exists
        if resume_mode and os.path.exists(output_path):
            print(f"⏩ [{i}/{total_rows}] Skipping {filename} (already exists)")
            skipped_count += 1
            generated_files.append(output_path)
            continue

        print(f"\n[{i}/{total_rows}] Processing: {filename}")
        
        try:
            result_file = generate_speech_direct(
                ref_audio_proc=ref_audio_proc,
                ref_text_proc=ref_text_proc,
                gen_text=transcript,
                speed=speed,
                output_dir=output_dir,
                output_file=filename,
            )
            generated_files.append(result_file)
        except Exception as e:
            print(f"⚠️  Failed to generate {filename}: {e}")
            error_count += 1
            continue

    # Summary
    success_count = len(generated_files) - skipped_count
    print(f"\n" + "=" * 60)
    print(f"🎉 COMPLETED!")
    print(f"   Total rows: {total_rows}")
    print(f"   Generated: {success_count} new files")
    print(f"   Skipped (existing): {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"   Output directory: {output_dir}")
    print("=" * 60)
    
    return generated_files


# ========================
# EXECUTION
# ========================
if __name__ == "__main__":
    start_time = time.time()

    print("=" * 60)
    print("🎤 CSV to WAV Converter (F5-TTS)")
    print("=" * 60)
    print(f"📄 Input CSV: {CSV_FILE}")
    print(f"📂 Output Dir: {OUTPUT_DIR}")
    print(f"🎙️  Reference: {REF_AUDIO}")
    print(f"⚙️  Model: {MODEL_NAME}")
    print(f"⚙️  Speed: {SPEED}x")
    print(f"🔄 Resume Mode: {RESUME_MODE}")
    print("=" * 60)

    # Validate input file
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: CSV file not found: {CSV_FILE}")
        sys.exit(1)

    # Preprocess voice once for efficiency
    cached_voice = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)

    # Generate all WAVs
    generated_files = csv_to_wavs_optimized(
        csv_file=CSV_FILE,
        output_dir=OUTPUT_DIR,
        preprocessed_voice=cached_voice,
        resume_mode=RESUME_MODE,
    )

    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"\n⏱️  Total time taken: {total_minutes:.2f} minutes ({total_minutes/60:.2f} hours)")
