import numpy as np
import soundfile as sf
import noisereduce as nr

def clean_vocals(input_file, output_file):
    """
    Clean vocals using noisereduce library - Windows friendly version.
    This avoids any permissions issues.
    """
    print(f"Loading: {input_file}")
    
    # Load the audio file
    audio, sample_rate = sf.read(input_file)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("Converting stereo to mono")
        audio = np.mean(audio, axis=1)
    
    # Find noise profile (assuming first 1 second is noise/silence)
    print("Extracting noise profile")
    noise_length = min(int(sample_rate), len(audio) // 4)
    noise_sample = audio[:noise_length]
    
    # Reduce noise
    print("Reducing noise...")
    reduced_noise = nr.reduce_noise(
        y=audio, 
        y_noise=noise_sample,
        sr=sample_rate,
        stationary=True,
        prop_decrease=0.75
    )
    
    # Normalize volume
    print("Normalizing volume")
    max_amp = np.max(np.abs(reduced_noise))
    if max_amp > 0:
        reduced_noise = reduced_noise * (0.9 / max_amp)
    
    # Save the result
    print(f"Saving to: {output_file}")
    sf.write(output_file, reduced_noise, sample_rate)
    print("Done!")

if __name__ == "__main__":
    # Set your file path
    input_output_file = "albiML2.wav"  # Using the same file for in/out

    # Run the cleaning 3 times
    for i in range(3):
        print(f"\n--- Pass {i+1}/3 ---")
        clean_vocals(input_output_file, input_output_file)
