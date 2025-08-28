import os
import torch
from transformers import pipeline
from pydub import AudioSegment
import soundfile as sf

def convert_to_wav(input_path):
    """Convert audio to WAV format if needed."""
    output_path = os.path.splitext(input_path)[0] + ".wav"

    # Convert non-WAV files
    if not input_path.lower().endswith(".wav"):
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"Converted {input_path} to {output_path}")

    return output_path

def load_model():
    """Load the ASR model."""
    model_path = "./whisper-finetuned"  # Ensure this exists
    return pipeline("automatic-speech-recognition", model=model_path, device=0 if torch.cuda.is_available() else -1)

def transcribe_audio(model, audio_path):
    """Transcribe the given audio file."""
    result = model(audio_path)
    return result["text"]

if __name__ == "__main__":
    audio_file = input("Enter the path to your audio file: ").strip()
    
    if not os.path.exists(audio_file):
        print("File not found. Please enter a valid file path.")
    else:
        wav_file = convert_to_wav(audio_file)
        model = load_model()
        print("Processing:", wav_file)
        transcription = transcribe_audio(model, wav_file)
        print("Transcription:", transcription)
