import sys
sys.path.append("src")

import numpy as np
from scipy.io import wavfile

from simple_vad_chunker import SimpleVADChunker, ChunkConfig
from whisper_asr import WhisperASR

# === LOAD AUDIO ===
wav_path = "harvard.wav"  
sr, audio = wavfile.read(wav_path)
print(f"Loaded '{wav_path}' with sample rate {sr} Hz, dtype {audio.dtype}")

# Convert to mono if stereo
if len(audio.shape) > 1:
    print("Stereo detected. Converting to mono...")
    audio = audio.mean(axis=1).astype(np.int16)

# Downsample to 16kHz if needed (optional: works best for Whisper)
if sr != 16000:
    from scipy.signal import resample_poly
    num_samples = int(len(audio) * 16000 / sr)
    audio = resample_poly(audio, 16000, sr).astype(np.int16)
    sr = 16000
    print(f"Resampled audio to {sr} Hz")

# === RUN VAD CHUNKER ===
chunk_cfg = ChunkConfig(sample_rate=sr)
chunker = SimpleVADChunker(chunk_cfg)
chunks = chunker.process_audio_buffer(audio)
print(f"Detected {len(chunks)} speech chunks:")

# === LOAD WHISPER ===
asr = WhisperASR("small")  # Load model once

# === TRANSCRIBE CHUNKS ===
for i, chunk in enumerate(chunks):
    result = asr.transcribe_chunk(chunk.audio_data, sample_rate=sr)
    print(f"Chunk {i+1:2d} [{chunk.start_ms}-{chunk.end_ms} ms]: '{result.text}' (confidence: {result.confidence:.2f})")

    nbest = asr.generate_nbest(chunk.audio_data, n=5)
    print("   N-best:", nbest)

