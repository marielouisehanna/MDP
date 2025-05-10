import librosa
import pyworld as pw
import soundfile as sf
import numpy as np

# --------------------
# 1. Configuration
# --------------------
SR = 44100          # Sampling rate
FRAME_PERIOD = 3.0  # in milliseconds
F0_FLOOR = 50
F0_CEIL = 800
# CORRECTION_AMOUNT: 1.0 = full correction, 0.0 = no correction
CORRECTION_AMOUNT = 0.5  

# --------------------
# 2. Helper Functions
# --------------------
def load_audio(path, sr=SR):
    """Load an audio file using librosa."""
    audio, _ = librosa.load(path, sr=sr)
    return audio

def analyze_audio(audio, sr=SR, frame_period=FRAME_PERIOD):
    """
    Analyze audio with WORLD: extract f0 contour, spectral envelope, and aperiodicity.
    Returns f0, spectral envelope, aperiodicity, and the time axis.
    """
    audio = audio.astype(np.float64)
    f0, t = pw.harvest(audio, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=frame_period)
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)
    return f0, sp, ap, t

def compute_median_pitch(f0):
    """
    Compute the median pitch from the f0 contour considering only voiced frames.
    This gives a robust estimate of the overall pitch level.
    """
    voiced = f0 > 0
    if np.sum(voiced) == 0:
        return 0
    return np.median(f0[voiced])

# --------------------
# 3. Main Processing
# --------------------
def main():
    # Load the singer's (reference) and your (user) audio files.
    singer_audio = load_audio("albi.wav", SR)
    user_audio = load_audio("albiML2.wav", SR)
    
    # Ensure both audio files are of the same length.
    min_len = min(len(singer_audio), len(user_audio))
    singer_audio = singer_audio[:min_len]
    user_audio = user_audio[:min_len]
    
    # WORLD analysis on both signals.
    f0_singer, sp_singer, ap_singer, t_singer = analyze_audio(singer_audio, SR, FRAME_PERIOD)
    f0_user, sp_user, ap_user, t_user = analyze_audio(user_audio, SR, FRAME_PERIOD)
    
    # Compute the median pitch for the singer and for you.
    median_singer = compute_median_pitch(f0_singer)
    median_user = compute_median_pitch(f0_user)
    print("Median singer pitch:", median_singer)
    print("Median user pitch:", median_user)
    
    if median_user == 0:
        print("Error: User's pitch could not be detected properly.")
        return
    
    # Compute the correction ratio to shift your median pitch toward the singer's.
    computed_ratio = median_singer / median_user
    print("Computed correction ratio (singer/user):", computed_ratio)
    

    correction_ratio = computed_ratio  
    
    # Generate a fully corrected pitch contour by applying the correction ratio to your pitch.
    f0_full_corrected = np.copy(f0_user)
    voiced = f0_user > 0
    f0_full_corrected[voiced] = f0_user[voiced] * correction_ratio

    # Blend your original pitch with the fully corrected pitch.
    # With CORRECTION_AMOUNT=1.0, you use the full corrected pitch.
    f0_corrected = np.copy(f0_user)
    f0_corrected[voiced] = ((1 - CORRECTION_AMOUNT) * f0_user[voiced] +
                            CORRECTION_AMOUNT * f0_full_corrected[voiced])
    
    # Print sample values to see the effect.
    print("Original f0 sample (first 10 voiced frames):", f0_user[voiced][:10])
    print("Fully corrected f0 sample (first 10 voiced frames):", f0_full_corrected[voiced][:10])
    print("Blended f0 sample (first 10 voiced frames):", f0_corrected[voiced][:10])
    
    # Ensure the number of frames in f0, spectral envelope (sp), and aperiodicity (ap) match.
    num_frames = min(len(f0_corrected), sp_user.shape[0])
    f0_corrected = f0_corrected[:num_frames]
    sp = sp_user[:num_frames, :]
    ap = ap_user[:num_frames, :]
    
    # Synthesize the output using WORLD with your spectral envelope and aperiodicity.
    synthesized_audio = pw.synthesize(f0_corrected, sp, ap, SR, FRAME_PERIOD)
    
    # Save the output audio.
    output_path = "k4.wav"
    sf.write(output_path, synthesized_audio, SR)
    print("Synthesis complete. Output saved to", output_path)

if __name__ == "__main__":
    main()
