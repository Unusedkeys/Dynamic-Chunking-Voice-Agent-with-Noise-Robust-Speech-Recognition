 
import sounddevice as sd
import numpy as np

def test_microphone():
    print("=== Microphone Test ===")
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            marker = " <- DEFAULT INPUT" if i == sd.default.device[0] else ""
            print(f"  {i}: {device['name']}{marker}")
    
    sample_rate = 16000
    duration = 3
    print(f"\nRecording {duration} seconds at {sample_rate}Hz...")
    print("🎤 Speak into your microphone NOW!")
    
    try:
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype=np.int16)
        sd.wait()
        
        # Calculate energy to check if speech was detected
        audio_float = audio[:, 0].astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(audio_float ** 2))
        max_amplitude = np.max(np.abs(audio_float))
        
        print(f"✓ Successfully recorded {len(audio)} samples")
        print(f"Audio range: {audio.min()} to {audio.max()}")
        print(f"RMS Energy: {energy:.6f}")
        print(f"Max Amplitude: {max_amplitude:.6f}")
        
        if energy > 0.01:
            print("✅ EXCELLENT - Microphone working perfectly!")
        elif energy > 0.003:
            print("✅ GOOD - Microphone working, speak a bit louder")
        elif energy > 0.001:
            print("⚠️ WEAK - Check microphone settings or speak louder")
        else:
            print("❌ PROBLEM - Very low signal, check microphone connection")
        
        return energy > 0.001
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        print("Check Windows microphone permissions in Settings > Privacy > Microphone")
        return False

if __name__ == "__main__":
    success = test_microphone()
    if not success:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check Windows Settings > Privacy > Microphone")
        print("2. Make sure microphone is set as default device")
        print("3. Try a different microphone or headset")
