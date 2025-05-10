# voice_enhancer.py
# A command-line tool to enhance voice recordings
# Usage: python voice_enhancer.py albiML.wav output.wav

import sys
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa
import soundfile as sf
import argparse

def enhance_voice(input_file, output_file):
    """
    Enhance a voice recording by applying various audio processing techniques
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save enhanced audio file
    """
    print(f"Reading audio file: {input_file}")
    
    # Load audio file with librosa (handles various formats)
    try:
        audio, sample_rate = librosa.load(input_file, sr=None, mono=False)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("Converting stereo to mono")
        audio = np.mean(audio, axis=0)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    print(f"Audio loaded: {len(audio)/sample_rate:.2f} seconds, {sample_rate} Hz")
    print("Applying voice enhancement...")
    
    # 1. Noise reduction using spectral gating
    # Simple noise reduction by spectral subtraction
    def reduce_noise(audio_data, sample_rate):
        # Get noise profile from the first 0.5 seconds (adjust as needed)
        noise_sample_length = int(sample_rate * 0.5)
        if len(audio_data) <= noise_sample_length:
            noise_sample_length = len(audio_data) // 4
        
        noise_sample = audio_data[:noise_sample_length]
        
        # Calculate noise spectrum
        n_fft = 2048
        noise_stft = librosa.stft(noise_sample, n_fft=n_fft)
        noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
        
        # Process the audio
        audio_stft = librosa.stft(audio_data, n_fft=n_fft)
        audio_power = np.abs(audio_stft)**2
        
        # Apply spectral gating
        gain = 1 - np.minimum(1, noise_power[:, np.newaxis] / (audio_power + 1e-10))
        gain = np.maximum(gain, 0.1)  # Minimum gain
        
        # Apply gain to original STFT
        enhanced_stft = audio_stft * gain
        
        # Inverse STFT
        enhanced_audio = librosa.istft(enhanced_stft, length=len(audio_data))
        
        return enhanced_audio
    
    # Apply noise reduction
    audio = reduce_noise(audio, sample_rate)
    
    # 2. Apply equalization for voice enhancement
    # Create filters using biquad filters
    def apply_eq(audio_data, sample_rate):
        # Custom bass boost (low shelf) implementation since scipy.signal.butter doesn't support gain
        def bass_boost(signal_data, sample_rate, cutoff=200, gain_db=6):
            # Convert gain from dB to linear
            gain = 10 ** (gain_db / 20)
            
            # Design a biquad low shelf filter
            w0 = 2 * np.pi * cutoff / sample_rate
            alpha = np.sin(w0) / 2 * np.sqrt(2)
            
            # Calculate filter coefficients
            b0 = gain * ((gain+1) - (gain-1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha)
            b1 = 2 * gain * ((gain-1) - (gain+1) * np.cos(w0))
            b2 = gain * ((gain+1) - (gain-1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha)
            a0 = (gain+1) + (gain-1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha
            a1 = -2 * ((gain-1) + (gain+1) * np.cos(w0))
            a2 = (gain+1) + (gain-1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha
            
            # Normalize by a0
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1/a0, a2/a0])
            
            # Apply the filter
            return signal.lfilter(b, a, signal_data)
        
        # Custom high shelf implementation
        def treble_boost(signal_data, sample_rate, cutoff=3000, gain_db=4):
            # Convert gain from dB to linear
            gain = 10 ** (gain_db / 20)
            
            # Design a biquad high shelf filter
            w0 = 2 * np.pi * cutoff / sample_rate
            alpha = np.sin(w0) / 2 * np.sqrt(2)
            
            # Calculate filter coefficients
            b0 = gain * ((gain+1) + (gain-1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha)
            b1 = -2 * gain * ((gain-1) + (gain+1) * np.cos(w0))
            b2 = gain * ((gain+1) + (gain-1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha)
            a0 = (gain+1) - (gain-1) * np.cos(w0) + 2 * np.sqrt(gain) * alpha
            a1 = 2 * ((gain-1) - (gain+1) * np.cos(w0))
            a2 = (gain+1) - (gain-1) * np.cos(w0) - 2 * np.sqrt(gain) * alpha
            
            # Normalize by a0
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1/a0, a2/a0])
            
            # Apply the filter
            return signal.lfilter(b, a, signal_data)
        
        # Voice presence - peaking filter at 2.5kHz
        def peaking_filter(signal_data, sample_rate, center_freq=2500, bandwidth=1000, gain_db=3):
            # Convert gain from dB to linear
            gain = 10 ** (gain_db / 20)
            
            # Design a biquad peaking filter
            w0 = 2 * np.pi * center_freq / sample_rate
            bandwidth_normalized = bandwidth / sample_rate
            alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * bandwidth_normalized * w0 / np.sin(w0))
            
            # Calculate filter coefficients
            b0 = 1 + alpha * gain
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * gain
            a0 = 1 + alpha / gain
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / gain
            
            # Normalize by a0
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1/a0, a2/a0])
            
            # Apply the filter
            return signal.lfilter(b, a, signal_data)
        
        # Apply bass boost
        audio_bass = bass_boost(audio_data, sample_rate, cutoff=200, gain_db=6)
        
        # Apply treble enhancement
        audio_treble = treble_boost(audio_bass, sample_rate, cutoff=3000, gain_db=4)
        
        # Apply presence boost (2-3kHz range to enhance speech clarity)
        audio_presence = peaking_filter(audio_data, sample_rate, center_freq=2500, bandwidth=1000, gain_db=5)
        
        # Mix presence with the processed signal
        enhanced = audio_treble + 0.3 * audio_presence
        
        # Additional warmth in mid-range
        vocal_warmth = peaking_filter(enhanced, sample_rate, center_freq=900, bandwidth=800, gain_db=2)
        
        return vocal_warmth
    
    # Apply EQ
    audio = apply_eq(audio, sample_rate)
    
    # 3. Compression - reduce dynamic range for more consistent volume
    def apply_compression(audio_data, threshold=-20, ratio=4, attack=0.005, release=0.1):
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold / 20)
        
        # Initialize gain memory
        gain_memory = 1.0
        compressed = np.zeros_like(audio_data)
        
        # Simple RMS-based compression
        for i in range(len(audio_data)):
            # Get sample amplitude
            amplitude = abs(audio_data[i])
            
            # Determine gain
            if amplitude > threshold_linear:
                # Above threshold, compress
                gain = threshold_linear + (amplitude - threshold_linear) / ratio
                gain = gain / amplitude if amplitude > 0 else 1.0
            else:
                # Below threshold, leave alone
                gain = 1.0
            
            # Apply attack/release smoothing
            if gain < gain_memory:
                # Use attack time for gain reduction
                gain_memory = attack * gain + (1 - attack) * gain_memory
            else:
                # Use release time for gain increase
                gain_memory = release * gain + (1 - release) * gain_memory
            
            # Apply gain
            compressed[i] = audio_data[i] * gain_memory
        
        # Apply makeup gain
        makeup_gain = 1.0 / (threshold_linear + (1 - threshold_linear) / ratio)
        compressed = compressed * makeup_gain * 0.7  # Reduce by 30% to prevent clipping
        
        return compressed
    
    # Apply compression
    audio = apply_compression(audio)
    
    # 4. Add subtle reverb for fullness
    def add_reverb(audio_data, sample_rate, reverb_level=0.1):
        # Create a simple room impulse response
        impulse_length = int(sample_rate * 0.5)  # 500ms reverb
        impulse = np.random.rand(impulse_length) * 2 - 1
        impulse = impulse * np.exp(-np.linspace(0, 10, impulse_length))
        impulse = impulse / np.sum(np.abs(impulse))
        
        # Convolve for reverb
        reverb_audio = signal.convolve(audio_data, impulse, mode='full')[:len(audio_data)]
        
        # Mix dry and wet signals
        result = (1 - reverb_level) * audio_data + reverb_level * reverb_audio
        
        return result
    
    # Apply subtle reverb
    audio = add_reverb(audio, sample_rate)
    
    # Final normalization
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Save the enhanced audio
    print(f"Saving enhanced audio to: {output_file}")
    sf.write(output_file, audio, sample_rate)
    print("Enhancement complete!")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enhance voice recordings')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('output_file', help='Path to save enhanced audio file')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Enhance the voice
    enhance_voice(args.input_file, args.output_file)

if __name__ == "__main__":
    main()