import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from resemblyzer import VoiceEncoder
import numpy as np
import librosa

# Define paths and configurations
CURRENT_DIR = os.getcwd()
RECORDINGS_DIR = os.path.join(CURRENT_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

SENTENCE = "Watan al nujoum... ana huna, haddi2... atzakar men ana? Alama7ta fi el madi el ba3id, fatan ghriran ar3ana?"

solfege_notes = {
    'Do': 261.63,  # C4
    'Re': 293.66,  # D4
    'Mi': 329.63,  # E4
    'Fa': 349.23,  # F4
    'Sol': 392.00, # G4
    'La': 440.00,  # A4
    'Si': 493.88   # B4
}

recording_files = {note: os.path.join(RECORDINGS_DIR, f"user_{note}.wav") for note in solfege_notes}
combined_audio_file = os.path.join(RECORDINGS_DIR, "ML3.wav")
embedding_file = os.path.join(RECORDINGS_DIR, "ML3.npy")

def record_note(note, filename, duration=3, samplerate=44100):
    """Record a single solfège note and save it to a file."""
    print(f"Recording {note}... Please sing now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio_data, samplerate)
    print(f"Saved: {filename}")

def record_all_notes():
    """Record all solfège notes (Do, Re, Mi, Fa, Sol, La, Si)."""
    for note, filename in recording_files.items():
        record_note(note, filename)

def record_sentence(filename, duration=10, samplerate=44100):
    """Record the spoken sentence"""
    print("\nNow please read this sentence:")
    print(SENTENCE)
    print("Recording starts now...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio_data, samplerate)
    print(f"Saved sentence recording to: {filename}")

def combine_audio():
    """Combine notes and sentence into one audio file"""
    combined = AudioSegment.silent(duration=500)
    
    # Add musical notes
    for note in solfege_notes:
        filepath = recording_files[note]
        if os.path.exists(filepath):
            sound = AudioSegment.from_wav(filepath)
            combined += sound + AudioSegment.silent(duration=300)
    
    # Add sentence recording
    sentence_path = os.path.join(RECORDINGS_DIR, "sentence.wav")
    if os.path.exists(sentence_path):
        combined += AudioSegment.silent(duration=1000)  # 1 second pause
        combined += AudioSegment.from_wav(sentence_path)
    
    combined.export(combined_audio_file, format="wav")
    print(f"Combined audio saved as {combined_audio_file}")

def detect_pitch(filename):
    """
    Detect the average pitch of an audio file using librosa.pyin.
    Adjust fmin and fmax to [C3, C6] (approx. 130Hz to 1046Hz) to reduce octave errors.
    """
    y, sr = librosa.load(filename, sr=44100)
    fmin = librosa.note_to_hz('C3')
    fmax = librosa.note_to_hz('C6')
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, pad_mode='constant')
    
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        print(f"No pitch detected in {filename}.")
        return None
    avg_pitch = np.median(valid_f0)
    
    if avg_pitch < 150:
        avg_pitch *= 2
    return avg_pitch

def map_pitch_to_note(pitch):
    """Map a detected pitch to the closest solfège note based on standard frequencies."""
    if pitch is None:
        return None
    return min(solfege_notes, key=lambda note: abs(solfege_notes[note] - pitch))

def analyze_recordings():
    """Analyze each individual recording: detect its pitch and map it to a solfège syllable."""
    print("Analyzing recorded notes...")
    for note, filename in recording_files.items():
        if os.path.exists(filename):
            pitch = detect_pitch(filename)
            mapped_note = map_pitch_to_note(pitch)
            if pitch is not None:
                print(f"{note}: Detected pitch = {pitch:.2f} Hz, Mapped to: {mapped_note}")
            else:
                print(f"{note}: No pitch detected.")
        else:
            print(f"File {filename} not found!")

def generate_embedding():
    if not os.path.exists(combined_audio_file):
        print(f"Error: {combined_audio_file} not found!")
        return
    print("Generating voice embedding...")
    wav, sr = librosa.load(combined_audio_file, sr=16000)
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)
    np.save(embedding_file, embedding)
    print(f"Voice embedding saved as {embedding_file}")



def evaluate_pitch_accuracy(filename, expected_note):
    """
    Analyze if the recorded note matches the expected solfège note.
    Provides simple feedback on pitch correction.
    """
    pitch = detect_pitch(filename)
    
    if pitch is None:
        return f"Could not detect your {expected_note}. Try singing louder."
    
    expected_freq = solfege_notes[expected_note]
    
    # Find which note they actually sang
    actual_note = map_pitch_to_note(pitch)
    
    # Determine if the pitch is accurate (within 5% is considered good)
    pitch_ratio = pitch / expected_freq
    
    if 0.97 <= pitch_ratio <= 1.03:
        return f"✅ Your {expected_note} sounds good!"
    
    # Simple feedback for non-musicians
    if actual_note == expected_note:
        if pitch > expected_freq:
            return f"⚠️ Your {expected_note} is slightly high"
        else:
            return f"⚠️ Your {expected_note} is slightly low" 
    else:
        # They sang a completely different note
        note_difference = list(solfege_notes.keys()).index(actual_note) - list(solfege_notes.keys()).index(expected_note)
        
        if note_difference > 0:
            return f"❌ You sang {actual_note} instead of {expected_note}. Try lower"
        else:
            return f"❌ You sang {actual_note} instead of {expected_note}. Try higher"

def analyze_recordings():
    """Analyze each recording with simple feedback."""
    print("Analyzing your notes...")
    correct_notes = 0
    
    for note, filename in recording_files.items():
        if os.path.exists(filename):
            pitch = detect_pitch(filename)
            actual_note = map_pitch_to_note(pitch)
            
            if pitch is not None:
                # First tell them what they actually sang
                print(f"{note}: You sang {actual_note} ({pitch:.0f} Hz)")
                
                # Then give feedback on how to improve
                feedback = evaluate_pitch_accuracy(filename, note)
                print(f"   {feedback}")
                
                if "✅" in feedback:
                    correct_notes += 1
            else:
                print(f"{note}: No sound detected")
        else:
            print(f"{note}: Recording not found")
    
    # Give overall feedback
    total_notes = len(solfege_notes)
    if correct_notes == total_notes:
        print("\n🎉 Perfect! All notes are on pitch!")
    elif correct_notes >= total_notes * 0.7:
        print(f"\n👍 Good job! You got {correct_notes} out of {total_notes} notes right")
    else:
        print(f"\n🎵 You got {correct_notes} out of {total_notes} notes right. Keep practicing!")

def main():
    print("Welcome to Ghanili Chwayi Chwayi!")
    print("Let's create your voice profile")
    
    # 1. Record musical notes
    print("\n🎵 Recording solfège notes")
    print("Sing each note when prompted")
    record_all_notes()
    
    # 2. Record spoken sentence
    print("\n🗣️ Now record a sentence")
    sentence_path = os.path.join(RECORDINGS_DIR, "sentence.wav")
    record_sentence(sentence_path)
    
    # 3. Analyze pitch accuracy
    print("\n📊 Analyzing your singing")
    analyze_recordings()
    
    # 4. Create voice profile
    print("\n🔄 Creating voice profile")
    combine_audio()
    generate_embedding()
    
    print("\n✅ Done! Your voice profile is ready")
    print("\nNow you can enhance your singing with our app!")

if __name__ == "__main__":
    main()