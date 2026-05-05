# main.py — Optimized for Large SRT Files
# No subprocess. Direct import. Model + vocoder loaded once. Preprocess once.

# 🔔 CHANGE THIS
# %%capture capture_output_hynt
srt_files = [
    "/content/input.srt",
    # "/content/nguyệt quang bảo kính_part004.ge25pro.fixedbySE.tts_clean.vn.srt",
    # "/content/nguyệt quang bảo kính_part005.ge25pro.fixedbySE.tts_clean.vn.srt",
    # "/content/nguyệt quang bảo kính_part006.ge25pro.fixedbySE.tts_clean.vn.srt",
] # chia tay bs Trình khóc: thuong/ha moi cai 5part, lam tiep tu thuong p2

import os
import sys
import time
import re
import shutil
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
from google.colab import files
import pysrt
from vinorm import TTSnorm

# Print lines that may cause TTS err OR contain at least one Chinese character
for srt_file in srt_files:
    print("-" * 40)
    print('SOME SUBTITLE LINES TO REVIEW:')
    print(f'({srt_file})')
    print("-" * 40)
    subs = pysrt.open(srt_file, encoding='utf-8')
    for i, sub in enumerate(subs, start=1):
        normalized_text = TTSnorm(sub.text).strip()

        # Check if normalized text is empty OR contains at least one Chinese character
        contains_chinese = bool(re.search(r'[\u4e00-\u9fff]', sub.text))

        if normalized_text == '' or contains_chinese:
            sound(jump_url)
            print(f"Index: {i}")
            print(f"Original text: {repr(sub.text)}")
            if contains_chinese:
                print("(Contains Chinese characters)")
            print("-" * 40)
print('(end of review)\n')

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
from f5_tts.model import DiT, UNetT  # needed for model config
from omegaconf import OmegaConf
import soundfile as sf
import numpy as np
from cached_path import cached_path

for srt_file in srt_files:
    try:
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
            model_name="F5TTS_Base",
            ckpt_file="",
            vocab_file="",
            vocoder_name="vocos",
            load_vocoder_from_local=False
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

                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))

            # Load model
            ema_model = load_model(
                model_cls,
                model_cfg.arch,
                ckpt_file,
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
                'ckpt_file': ckpt_file,
                'model_name': model_name,
            })

            print(f"✅ Model {model_name} loaded and cached.")
            return ema_model, vocoder


        # ========================
        # TEXT PROCESSING
        # ========================
        def post_process(text):
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
            speed=1.0,
            vocoder_name="vocos",
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            fix_duration=None,
            output_dir="./output",
            output_file="output.wav",
            remove_silence=False,
            save_chunk=False,
            chunk_dir=None,
            vocab_file="",
            ckpt_file="",
            model_name="F5TTS_Base",
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

            print("🚀 Running F5-TTS inference (direct)...")
            print(f"→ Reference text: {ref_text_proc[:100]}...")
            print(f"→ Generated text: {gen_text_clean}")

            try:
                audio_segment, final_sample_rate, _ = infer_process(
                    ref_audio_proc,
                    ref_text_proc,
                    gen_text_clean,
                    ema_model,
                    vocoder,
                    mel_spec_type=vocoder_name,
                    target_rms=target_rms,
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    fix_duration=fix_duration,
                )

                # Save audio
                with open(wave_path, "wb") as f:
                    sf.write(f.name, audio_segment, final_sample_rate)
                    if remove_silence:
                        remove_silence_for_generated_wav(f.name)
                    print(f"✅ Audio saved to: {wave_path}")

                # Optional: save chunk
                if save_chunk and chunk_dir:
                    os.makedirs(chunk_dir, exist_ok=True)
                    chunk_name = f"{len(os.listdir(chunk_dir))}_{gen_text_clean[:50].replace(' ', '_')}.wav"
                    chunk_path = os.path.join(chunk_dir, chunk_name)
                    sf.write(chunk_path, audio_segment, final_sample_rate)

                return str(wave_path)

            except Exception as e:
                print(f"❌ Error during direct inference: {e}")
                raise RuntimeError(f"F5-TTS direct inference failed: {e}") from e


        # ========================
        # SRT PROCESSING (OPTIMIZED)
        # ========================
        def estimate_speech_duration(text, cps=12.0):
            if not text:
                return 0.0
            text_clean = text.strip()
            num_chars = len(text_clean)
            duration_sec = num_chars / cps
            return duration_sec


        def parse_srt(srt_path):
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            pattern = re.compile(
                r'(\d+)\s*\n'
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n'
                r'(.*?)(?=\n\n|\Z)',
                re.DOTALL
            )

            entries = []
            for match in pattern.finditer(content):
                index = int(match.group(1))
                start = match.group(2)
                end = match.group(3)
                text = match.group(4).replace('\n', ' ').strip()
                entries.append({
                    'index': index,
                    'start': start,
                    'end': end,
                    'text': text
                })
            return entries


        def time_str_to_milliseconds(time_str):
            h, m, rest = time_str.split(':')
            s, ms = rest.split(',')
            return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


        # ========================
        # MAIN SRT → WAV PIPELINE
        # ========================
        def srt_to_wavs_optimized(
            srt_file,
            model_name="F5TTS_Base",
            ref_audio="",
            ref_text="",
            preprocessed_voice=None,
            speed=1.0,
            max_speed=2.5,
            cps=12.0,
            vocoder_name="vocos",
            vocab_file="",
            ckpt_file="",
            output_dir="./output",
            naming_scheme="index",
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            fix_duration=None,
            remove_silence=False,
            save_chunks=False,
        ):
            """
            Convert SRT to WAVs with DYNAMIC SPEED (unless max_speed=1.0), using direct inference.
            Preprocess ref ONCE. Load model ONCE. Resume-safe.
            """
            print(f'[srt_to_wavs_optimized], max_speed={max_speed}')
            os.makedirs(output_dir, exist_ok=True)
            entries = parse_srt(srt_file)
            print(f"📁 Parsed {len(entries)} subtitle entries from {srt_file}")

            # Preprocess reference ONCE
            if preprocessed_voice is None:
                print("🎙️ Preprocessing reference voice...")
                ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)
            else:
                print("⚡ Using cached reference voice")
                ref_audio_proc, ref_text_proc = preprocessed_voice

            # Prepare chunk dir
            chunk_dir = os.path.join(output_dir, "chunks") if save_chunks else None

            generated_files = []

            # If max_speed is 1.0, we ignore timing and use fixed speed
            use_dynamic_speed = max_speed != 1.0

            for entry in entries:
                idx = entry['index']
                text = entry['text']

                # Output filename
                if naming_scheme == "timestamp":
                    safe_time = entry['start'].replace(":", "_").replace(",", "_")
                    output_file = f"{safe_time}.wav"
                else:
                    output_file = f"{idx:03d}.wav"

                output_path = os.path.join(output_dir, output_file)

                # ✅ Resume: skip if exists
                if os.path.exists(output_path):
                    print(f"⏩ Skipping line {idx}, already exists: {output_file}")
                    generated_files.append(output_path)
                    continue

                # Compute speed only if needed
                final_speed = speed
                if use_dynamic_speed:
                    start_ms = time_str_to_milliseconds(entry['start'])
                    end_ms = time_str_to_milliseconds(entry['end'])
                    duration_ms = end_ms - start_ms
                    duration_sec = duration_ms / 1000.0

                    natural_duration_sec = estimate_speech_duration(text, cps=cps)
                    required_speed = natural_duration_sec / duration_sec if duration_sec > 0 else speed
                    final_speed = min(max(speed, required_speed), max_speed)

                    if required_speed > max_speed:
                        print(f"⚠️  Line {idx} needs speed {required_speed:.2f}x → capped at {max_speed}x")

                    print(f"\n🔊 Line {idx}: '{text}'")
                    print(f"   Allowed: {duration_sec:.2f}s | Natural: ~{natural_duration_sec:.2f}s → Speed: {final_speed:.2f}x")
                else:
                    print(f"\n🔊 Line {idx}: '{text}' (fixed speed: {speed}x)")

                try:
                    result_file = generate_speech_direct(
                        ref_audio_proc=ref_audio_proc,
                        ref_text_proc=ref_text_proc,
                        gen_text=text,
                        speed=final_speed,
                        vocoder_name=vocoder_name,
                        target_rms=target_rms,
                        cross_fade_duration=cross_fade_duration,
                        nfe_step=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        fix_duration=fix_duration,
                        output_dir=output_dir,
                        output_file=output_file,
                        remove_silence=remove_silence,
                        save_chunk=save_chunks,
                        chunk_dir=chunk_dir,
                        vocab_file=vocab_file,
                        ckpt_file=ckpt_file,
                        model_name=model_name,
                    )
                    generated_files.append(result_file)
                except Exception as e:
                    print(f"⚠️  Failed to generate audio for subtitle #{idx}: {e}")
                    continue

            print(f"\n🎉 Completed. Generated {len(generated_files)} audio files in {output_dir}")
            return generated_files


        # ========================
        # MERGE WITH SILENCE
        # ========================
        def merge_wavs_with_silence(
            srt_file,
            wav_dir="./output",
            output_file="./output/merged_final.wav",
            naming_scheme="index",
            pad_start_ms=0,
            pad_end_ms=0
        ):
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            entries = parse_srt(srt_file)
            if not entries:
                raise ValueError("No entries found in SRT file.")

            entries.sort(key=lambda x: x['index'])
            audio_segments = []

            for entry in entries:
                idx = entry['index']
                start_ms = time_str_to_milliseconds(entry['start'])
                end_ms = time_str_to_milliseconds(entry['end'])

                if naming_scheme == "timestamp":
                    safe_time = entry['start'].replace(":", "_").replace(",", "_")
                    wav_file = os.path.join(wav_dir, f"{safe_time}.wav")
                else:
                    wav_file = os.path.join(wav_dir, f"{idx:03d}.wav")

                if not os.path.exists(wav_file):
                    print(f"⚠️  Missing WAV file: {wav_file}. Skipping this segment.")
                    continue

                audio = AudioSegment.from_wav(wav_file)
                audio_segments.append({
                    'index': idx,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_ms': len(audio),
                    'audio': audio,
                    'filename': wav_file
                })

            if not audio_segments:
                raise RuntimeError("No valid audio segments to merge.")

            final_audio = AudioSegment.silent(duration=pad_start_ms)
            prev_end_ms = pad_start_ms

            for seg in audio_segments:
                current_start_ms = seg['start_ms']
                silence_needed_ms = current_start_ms - prev_end_ms

                if silence_needed_ms > 0:
                    final_audio += AudioSegment.silent(duration=silence_needed_ms)
                    prev_end_ms = current_start_ms

                final_audio += seg['audio']
                prev_end_ms += seg['duration_ms']
                print(f"✅ Added segment #{seg['index']} from {seg['filename']} at ~{current_start_ms}ms")

            if pad_end_ms > 0:
                final_audio += AudioSegment.silent(duration=pad_end_ms)

            final_audio.export(output_file, format="wav")
            print(f"\n🎉 Merged audio saved to: {output_file}")
            print(f"   Total duration: {len(final_audio) / 1000:.2f} seconds")
            return output_file


        # ========================
        # FULL PIPELINE
        # ========================
        def srt_to_merged_wav_optimized(
            srt_file,
            model_name="F5TTS_Base",
            ref_audio="",
            ref_text="",
            preprocessed_voice=None,
            speed=1.0,
            max_speed=2.5,
            cps=12.0,
            vocoder_name="vocos",
            vocab_file="",
            ckpt_file="",
            temp_dir="./temp_wavs",
            final_output="./output/final.wav",
            naming_scheme="index",
            pad_start_ms=0,
            pad_end_ms=0,
            overwrite_final=False,
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            fix_duration=None,
            remove_silence=False,
            save_chunks=False,
        ):
            """
            Full optimized pipeline: SRT → WAVs → merged WAV.
            Resume-safe. Model + ref preprocessed once.
            """
            if os.path.exists(final_output) and not overwrite_final:
                print(f"⏩ Skipping: FINAL MERGED FILE ALREADY EXISTS → {final_output}")
                return final_output

            # Step 1: Generate individual wavs
            generated_files = srt_to_wavs_optimized(
                srt_file=srt_file,
                model_name=model_name,
                ref_audio=ref_audio,
                ref_text=ref_text,
                preprocessed_voice=preprocessed_voice,
                speed=speed,
                max_speed=max_speed,
                cps=cps,
                vocoder_name=vocoder_name,
                vocab_file=vocab_file,
                ckpt_file=ckpt_file,
                output_dir=temp_dir,
                naming_scheme=naming_scheme,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                fix_duration=fix_duration,
                remove_silence=remove_silence,
                save_chunks=save_chunks,
            )

            if not generated_files:
                print("⚠️ No WAV files to merge, skipping.")
                return None

            # Step 2: Merge wavs
            print(f"\n🔗 Merging {len(generated_files)} WAVs into final output...")
            merged_file = merge_wavs_with_silence(
                srt_file=srt_file,
                wav_dir=temp_dir,
                output_file=final_output,
                naming_scheme=naming_scheme,
                pad_start_ms=pad_start_ms,
                pad_end_ms=pad_end_ms,
            )

            return merged_file


        # ========================
        # EXECUTION
        # ========================
        if __name__ == "__main__":
            start_time = time.time()

            # Preprocess voice once
            cached_voice = preprocess_ref_audio_text(
                "/content/drive/MyDrive/shared_rakion/voices/Đây là bằng hữu của ta TRIMMED.wav",
                "Đây là bằng hữu của ta. Vị cô nương này không được khỏe. Các ngươi mau đưa nàng ấy vào trong chữa trị đi"

                # "/content/drive/MyDrive/shared_rakion/voices/cả 2 bên hãy cố gắng hiểu cho nhau.wav",
                # "cả 2 bên hãy cố gắng hiểu cho nhau"

                # "/content/drive/MyDrive/shared_rakion/voices/Lâm Giai cười cười mắt phượng liếc xéo TRIMMED.wav",
                # "Lâm Giai cười cười, mắt phượng liếc xéo, ánh mắt lưu chuyển, đôi môi đỏ vô cùng gợi cảm thực là quyến rũ. Lúc này chiếc xe Toyota cũng đã xuống ven đường, vị trí lái xe lộ ra một khuôn mặt quen thuộc, chính là bạn học cũ Lưu Vân Chí."

                # "/content/drive/MyDrive/shared_rakion/voices/Mày muốn chết hả.wav",
                # "Mày muốn chết hả? Đậu xanh rau má con mẹ nhà mày!"

                # "/content/drive/MyDrive/shared_rakion/voices/Em biết em sai rồi.wav",
                # "Em biết em sai rồi. Anh hãy ra đây đi. Em xin anh đấy!"

                # "/content/drive/MyDrive/shared_rakion/voices/Anh mạnh lên. Nữa đi.wav", #CUI
                # "Anh mạnh lên. Nữa đi, nữa đi, đừng dừng lại!"

                # "/content/drive/MyDrive/shared_rakion/voices/11labs - Nhưng tại sao anh không tự mình đến gặp em - Anna Sokolova 2.mp3",
                # "Nhưng tại sao anh không tự mình đến gặp em? Vẫn còn hận em sao? Giang Yến."

                # "/content/drive/MyDrive/shared_rakion/voices/Ngày 13 tháng 5 ngày mà Nini....wav",
                # "Ngày mười ba tháng năm. Chính là ngày này. Nini từ trong nhà đi ra bị một chiếc xe hơi màu đen tông phải. Mình đã quay ngày mà Ni."

                # "/content/drive/MyDrive/shared_rakion/voices/Sở tổng Sở tổng.wav", #SPD 1 CUNG CUI
                # "Sở tổng, Sở tổng, Sở tổng, cổ đông Ni Cô là mẹ đơn thân, không có khả năng tiếp quản Sở thị. Cô có cảm nghĩ gì? Con của cô là từ đâu mà ra? Bố của đứa trẻ là ai? Xin tiết lộ chút! Cô có liên hôn với Vương thị không? Xin nhường đường! Sở tổng, xin hãy phản hồi. Sở tổng, Sở tổng."

                # "/content/drive/MyDrive/shared_rakion/voices/Tin từ đài truyền hình.wav",
                # "Tin từ đài truyền hình, phòng tập kinh trung bị bỏ hoang, bất ngờ xảy ra hỏa hoạn. Hiện tại số người thương vong chưa rõ, lực lượng cứu hỏa đang nỗ lực cứu hộ."

                # "/content/drive/MyDrive/shared_rakion/voices/Ngay cả vì tiền thuốc men của con gái (sieu nhanh, temp 2).wav",
                # "Ngay cả vì tiền thuốc men của con gái. mà khiêng quan tài năm năm. bị mọi người cười nhạo. vẫn cam tâm tình nguyện. Giờ xem ra. Tô Ngữ vốn dĩ. không thèm để mắt đến căn nhà này."

                # "/content/Lúc Tô Ngữ kết hôn với tôi.wav",
                # "Lúc Tô Ngữ kết hôn với tôi. khăng khăng trên sổ đỏ, chỉ để tên của tôi, Cô ấy nói chỉ có như vậy, mới là sự đảm bảo lớn nhất cho tôi sau hôn nhân"

                # "/content/drive/MyDrive/shared_rakion/voices/Tôi ngốc đến mức tưởng rằng.wav",  # 'mono hơn' nè
                # "Tôi ngốc đến mức tưởng rằng. mình đã gặp được tình yêu đích thực. Bỏ ngoài tai sự phản đối kịch liệt của bố mẹ. không tiếc bỏ nhà ra đi."

                # "/content/drive/MyDrive/shared_rakion/voices/Ngay cả vì tiền thuốc men của con gái.wav",
                # "Ngay cả vì tiền thuốc men của con gái. mà khiêng quan tài năm năm. bị mọi người cười nhạo. vẫn cam tâm tình nguyện. Giờ xem ra. Tô Ngữ vốn dĩ. không thèm để mắt đến căn nhà này."

                # gen Ngày 13 from Ngày 13 raw
                # "/content/Em đã đặc biệt theo thói quen của anh [gen Ngày 13 from raw Ngày 13].wav",
                # "Em đã đặc biệt theo thói quen của anh"

                # gen Ngày 13 from Ngày 13 raw (Slower)
                # "/content/Dự án người nhân tạo [gen Ngày 13 from raw, Slower].wav",
                # "Dự án người nhân tạo"

                # "/content/Thư ký Lưu chỉ cần hầu hạ tốt Bùi tổng là có thể một bước lên mây.wav",
                # "Thư ký Lưu chỉ cần hầu hạ tốt Bùi tổng là có thể một bước lên mây"

                # "/content/Có một anh chàng mê ăn cơm gà.wav",  # CUI
                # "Có một anh chàng mê ăn cơm gà đến mức mỗi ngày đều ăn. Một hôm, anh đi vào quán mới mở. Vừa ngồi xuống, anh nói với cô phục vụ: – Cho anh một dĩa cơm gà! Cô phục vụ mỉm cười: – Dạ, quán em chuyên bán cơm sườn ạ. Anh chàng gãi đầu, ngẫm nghĩ một chút, rồi nói: – Vậy cho anh một dĩa… cơm gà, nhưng thay gà bằng sườn nha! "

#                 "/content/Ở một cái xóm nhỏ miền Tây, có ông Ba Tèo nổi tiếng ham cá độ đá gà.wav",
#                 r"""Ở một cái xóm nhỏ miền Tây, có ông Ba Tèo nổi tiếng ham cá độ đá gà.
# Một bữa, vợ ông la dữ lắm:
# – Ông mà còn đi đá gà nữa là tui bỏ về ngoại đó nghe!
# Ông Ba nghe xong, cũng hứa hẹn:
# – Thôi bả ơi, tui thề… từ nay hổng có đá gà nữa!"""

#                 "/content/Nói xong, ổng đem hết tiền cất vô hũ gạo.wav",
#                 r"""Nói xong, ổng đem hết tiền cất vô hũ gạo, tự nhủ: “Mình tu tỉnh rồi, ở nhà phụ vợ lành mạnh thôi.”
# Mới được hai ngày, bạn nhậu rủ:
# – Ba ơi, có trường mới mở, gà chiến dữ lắm, đi coi hông?
# Ông Ba cắn răng:
# – Thôi, hổng đi, vợ cấm rồi."""

#                 "/content/Ở một cái xóm nhỏ miền Tây TRIMMED.wav",
#                 r"""Ở một cái xóm nhỏ miền Tây, có ông Ba Tèo nổi tiếng ham cá độ đá gà.
# Một bữa, vợ ông la dữ lắm:
# – Ông mà còn đi đá gà nữa là tui bỏ về ngoại đó nghe!
# Ông Ba nghe xong, cũng hứa hẹn:
# – Thôi bả ơi, tui thề… từ nay hổng có đá gà nữa!"""

                # "/content/drive/MyDrive/shared_rakion/voices/bbbb.wav",
                # "cccc"

            )

            # final_output = "bbbb"
            final_output = str(Path("./output") / Path(srt_file).with_suffix(".wav").name)

            # Generate & merge
            final_audio = srt_to_merged_wav_optimized(
                srt_file=srt_file,
                preprocessed_voice=cached_voice,
                temp_dir="./temp_wavs",
                # final_output="./output/Hôm qua mọi thứ ví như mây khói_part002.ge25pro.vn.merged_overlapfixed.wav",
                final_output=final_output,
                overwrite_final=False,
                naming_scheme="index",
                vocab_file="/content/drive/MyDrive/shared_underwater/F5-TTS-Vietnamese-ViVoice_models/vocab.txt",
                ckpt_file="/content/drive/MyDrive/shared_underwater/F5-TTS-Vietnamese-ViVoice_models/model_last.pt",
                model_name="F5TTS_Base",
                vocoder_name="vocos",
                remove_silence=False,
                save_chunks=False,    # optional
                max_speed=1.0,  # capped speaking speed, default=2.5
            )

            print(f"🎬 Final dubbed audio ready: {final_audio}")
            end_time = time.time()
            total_minutes = (end_time - start_time) / 60
            print(f"⏱️  Total time taken: {total_minutes:.2f} minutes")

            # DOWNLOAD FINAL WAV AND ALSO TEMP_WAVS
            new_temp_wavs_name = f"/content/temp_wavs_{Path(srt_file).name.split('.')[0]}"
            os.rename('/content/temp_wavs', new_temp_wavs_name)
            zip_base_name = Path(new_temp_wavs_name).name  # e.g., "temp_wavs_filename"
            zip_output_path = f"/content/{zip_base_name}"  # e.g., "/content/temp_wavs_filename"
            zip_path = shutil.make_archive(
                base_name=zip_output_path,     # Output: /content/temp_wavs_filename.zip
                format='zip',                  # Archive format
                root_dir='/content',           # Parent directory of the folder to zip
                base_dir=Path(new_temp_wavs_name).name  # Folder name to put at top level of zip
            )

            copy_to_mydrive(final_output)  # copy_to_mydrive auto-detects dir or file
            copy_to_mydrive(zip_path)
            sound()
    except Exception as e:
          print(e)
          sound(good_day_to_die_url)
