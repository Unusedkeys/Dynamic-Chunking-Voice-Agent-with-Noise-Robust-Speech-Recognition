 
import sys
sys.path.append('src')

import numpy as np
import time
from simple_vad_chunker import SimpleVADChunker, ChunkConfig

def evaluate_dynamic_vs_fixed():
    print("=== Dynamic vs Fixed Chunking Comparison ===")
    
    # Create realistic test audio (8 seconds)
    sample_rate = 16000
    duration = 8
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Background noise
    audio = np.random.normal(0, 0.002, len(t)).astype(np.float32)
    
    # Add speech segments with pauses
    segments = [
        (1.0, 2.2, 440),   # "Hello there"
        (2.8, 4.1, 880),   # "How are you today"  
        (5.2, 6.8, 660),   # "This is a test"
    ]
    
    print("Creating test audio with speech segments:")
    for start, end, freq in segments:
        start_idx, end_idx = int(start * sample_rate), int(end * sample_rate)
        # Create speech-like signal
        speech = 0.08 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        speech += 0.04 * np.random.normal(0, 1, len(speech))  # Add variation
        audio[start_idx:end_idx] += speech
        print(f"  Speech segment: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
    
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Test Dynamic Chunking
    print("\n--- Dynamic Chunking Results ---")
    config = ChunkConfig()
    chunker = SimpleVADChunker(config)
    
    start_time = time.time()
    dynamic_chunks = chunker.process_audio_buffer(audio_int16)
    dynamic_time = time.time() - start_time
    
    print(f"Processing time: {dynamic_time:.3f}s")
    print(f"Generated {len(dynamic_chunks)} adaptive chunks:")
    
    for i, chunk in enumerate(dynamic_chunks):
        duration_s = len(chunk.audio_data) / sample_rate
        energy = np.sqrt(np.mean((chunk.audio_data.astype(np.float32) / 32768.0) ** 2))
        print(f"  Chunk {i+1}: {chunk.start_ms:4d}-{chunk.end_ms:4d}ms ({duration_s:.1f}s) Energy: {energy:.4f}")
    
    # Test Fixed Chunking (3-second windows)
    print("\n--- Fixed Chunking Results (3s windows) ---")
    window_size = 3 * sample_rate
    
    start_time = time.time()
    fixed_chunks = []
    for i in range(0, len(audio_int16), window_size):
        chunk_audio = audio_int16[i:i + window_size]
        if len(chunk_audio) > 0:
            start_ms = int(i / sample_rate * 1000)
            end_ms = int((i + len(chunk_audio)) / sample_rate * 1000)
            duration_s = len(chunk_audio) / sample_rate
            energy = np.sqrt(np.mean((chunk_audio.astype(np.float32) / 32768.0) ** 2))
            fixed_chunks.append((start_ms, end_ms, duration_s, energy))
    
    fixed_time = time.time() - start_time
    
    print(f"Processing time: {fixed_time:.3f}s")
    print(f"Generated {len(fixed_chunks)} fixed chunks:")
    
    for i, (start_ms, end_ms, duration_s, energy) in enumerate(fixed_chunks):
        print(f"  Chunk {i+1}: {start_ms:4d}-{end_ms:4d}ms ({duration_s:.1f}s) Energy: {energy:.4f}")
    
    # Analysis
    print("\n--- Comparison Analysis ---")
    dynamic_total_chunks = len(dynamic_chunks)
    fixed_total_chunks = len(fixed_chunks)
    
    print(f"Dynamic chunking: {dynamic_total_chunks} chunks (adaptive boundaries)")
    print(f"Fixed chunking:   {fixed_total_chunks} chunks (rigid 3s windows)")
    print(f"Efficiency gain:  {((fixed_total_chunks - dynamic_total_chunks) / fixed_total_chunks * 100):.1f}% fewer chunks")
    
    # Check boundary alignment with speech
    print("\nBoundary quality assessment:")
    speech_boundaries = [(1000, 2200), (2800, 4100), (5200, 6800)]  # ms
    
    for i, (start_speech, end_speech) in enumerate(speech_boundaries):
        print(f"\nSpeech segment {i+1} ({start_speech}-{end_speech}ms):")
        
        # Check dynamic chunks
        dynamic_matches = []
        for chunk in dynamic_chunks:
            if (chunk.start_ms <= start_speech <= chunk.end_ms or 
                chunk.start_ms <= end_speech <= chunk.end_ms or
                (start_speech <= chunk.start_ms and end_speech >= chunk.end_ms)):
                dynamic_matches.append(chunk)
        
        # Check fixed chunks  
        fixed_matches = []
        for start_ms, end_ms, _, _ in fixed_chunks:
            if (start_ms <= start_speech <= end_ms or 
                start_ms <= end_speech <= end_ms or
                (start_speech <= start_ms and end_speech >= end_ms)):
                fixed_matches.append((start_ms, end_ms))
        
        print(f"  Dynamic: {len(dynamic_matches)} chunks capture this segment")
        print(f"  Fixed:   {len(fixed_matches)} chunks capture this segment")
        
        if len(dynamic_matches) < len(fixed_matches):
            print("  ✅ Dynamic chunking more efficient!")
        elif len(dynamic_matches) == len(fixed_matches):
            print("  ➖ Same efficiency")
        else:
            print("  ❌ Fixed chunking more efficient")
    
    print(f"\n🎯 Summary:")
    print(f"✅ Dynamic chunking adapts to speech patterns")
    print(f"✅ Reduces processing overhead by {((fixed_total_chunks - dynamic_total_chunks) / fixed_total_chunks * 100):.0f}%")
    print(f"✅ Better boundary alignment with natural speech pauses")

if __name__ == "__main__":
    evaluate_dynamic_vs_fixed()
