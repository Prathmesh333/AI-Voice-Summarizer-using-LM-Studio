import os
import time
import gradio as gr
import openai
import whisper
from datetime import datetime
import shutil
import scipy.io.wavfile as wavfile
import numpy as np
import traceback
from typing import Literal, Optional

# --- Create Necessary Directories ---
os.makedirs("audio", exist_ok=True)
os.makedirs("summaries", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- LM Studio Configuration ---
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL_ID = "deepseek-r1-distill-qwen-7b"

# --- Whisper Configuration ---
WHISPER_MODEL_PATH = "./models"
AVAILABLE_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
DEFAULT_WHISPER_MODEL_SIZE = "base"

# --- Global variables ---
loaded_whisper_models = {}
client = None

# --- Model Loading ---
def load_whisper_model_by_size(size):
    """Loads a specific Whisper model size, caching it if successful."""
    global loaded_whisper_models

    if size not in AVAILABLE_WHISPER_MODELS:
        print(f"Error: Invalid Whisper model size requested: {size}")
        return None, f"Invalid model size: {size}"

    # Return cached model if already loaded
    if size in loaded_whisper_models:
        print(f"Using cached Whisper model: '{size}'")
        return loaded_whisper_models[size], None

    # Load the model if not cached
    print(f"Loading Whisper model '{size}' from '{WHISPER_MODEL_PATH}'...")
    print("(This might take time, especially for larger models or first download)")
    try:
        start_time = time.time()
        model = whisper.load_model(size, download_root=WHISPER_MODEL_PATH)
        end_time = time.time()
        print(f"Whisper model '{size}' loaded successfully in {end_time - start_time:.2f} seconds.")
        loaded_whisper_models[size] = model
        return model, None
    except Exception as e:
        error_msg = f"Error loading Whisper model '{size}': {e}"
        print(error_msg)
        print("Please ensure 'openai-whisper' is installed, 'ffmpeg' is available,")
        print(f"the model can be downloaded to '{WHISPER_MODEL_PATH}', and you have enough RAM/VRAM.")
        return None, error_msg

def initialize_models():
    """Initialize LM Studio client and load the default Whisper model."""
    global client

    # Initialize OpenAI Client for LM Studio
    try:
        client = openai.OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )
        print(f"OpenAI client configured for LM Studio at {LM_STUDIO_BASE_URL}")
    except Exception as e:
        print(f"Error initializing OpenAI client for LM Studio: {e}")
        client = None

    # Load the default Whisper Model at startup
    print(f"\nLoading default Whisper model ('{DEFAULT_WHISPER_MODEL_SIZE}')...")
    model, error = load_whisper_model_by_size(DEFAULT_WHISPER_MODEL_SIZE)
    if error:
        print(f"Failed to load default Whisper model: {error}")

# --- Audio/Text Processing Functions ---
def save_audio_file(audio_data):
    """Save uploaded or recorded audio to the audio folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            file_ext = os.path.splitext(audio_data)[1]
            source_type = "recorded_or_uploaded"
            filename = f"audio/{source_type}_{timestamp}{file_ext if file_ext else '.wav'}"
            print(f"Copying/Moving audio from temp path {audio_data} to: {filename}")
            shutil.copy(audio_data, filename)
            return filename
        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, data = audio_data
            if isinstance(sample_rate, int) and isinstance(data, np.ndarray):
                filename = f"audio/recorded_{timestamp}.wav"
                print(f"Saving recorded audio tuple to: {filename}")
                if data.dtype == np.float32 or data.dtype == np.float64:
                     data = (data * 32767).astype(np.int16)
                elif data.dtype != np.int16 and data.dtype != np.uint8:
                     print(f"Warning: Unexpected numpy array dtype {data.dtype}. Attempting to convert to int16.")
                     data = data.astype(np.int16)
                wavfile.write(filename, sample_rate, data)
                return filename
            else:
                print(f"Warning: Unexpected data types in audio tuple: rate={type(sample_rate)}, data={type(data)}. Cannot save.")
                return None
        else:
            print(f"Warning: Unexpected audio data type: {type(audio_data)}. Content: {str(audio_data)[:100]}. Cannot save.")
            return None
    except Exception as e:
        print(f"Error saving audio file: {e}")
        traceback.print_exc()
        return None

def transcribe_audio(audio_path, model_size):
    """Transcribe audio using the specified Whisper model size, forcing English"""
    whisper_model, load_error = load_whisper_model_by_size(model_size)

    if load_error:
        return f"Error: Could not load Whisper model '{model_size}'. {load_error}"
    if not whisper_model:
         return f"Error: Failed to retrieve Whisper model '{model_size}' after load attempt."

    if not audio_path or not os.path.exists(audio_path):
        time.sleep(0.5) # Brief pause for filesystem sync
        if not audio_path or not os.path.exists(audio_path):
            return f"Error: Audio file not found at path: {audio_path}"

    try:
        print(f"Starting English transcription for: {audio_path} using '{model_size}' model...")
        start_time = time.time()
        result = whisper_model.transcribe(audio_path, language='en', fp16=False)
        transcript = result["text"].strip()
        end_time = time.time()
        print(f"Transcription finished in {end_time - start_time:.2f} seconds.")
        return transcript if transcript else "[No speech detected]"
    except Exception as e:
        error_msg = f"Error during transcription with model '{model_size}': {str(e)}"
        print(error_msg)
        if "ffmpeg" in str(e).lower():
             error_msg += "\n(Ensure ffmpeg is installed and in your system's PATH)"
        traceback.print_exc()
        return error_msg

def summarize_text(text, transcript_model_size="N/A"):
    """Summarize text using LM Studio, ensuring English output"""
    global client

    if not client:
        return "Error: LM Studio client not initialized. Please check console for details.", None

    if not text or not text.strip() or text.startswith("Error:") or text == "[No speech detected]":
        print(f"Skipping summarization for invalid/empty text: '{str(text)[:100]}...'")
        if text == "[No speech detected]":
             return "Cannot summarize: No speech was detected in the audio.", None
        elif text and text.startswith("Error:"):
             return f"Cannot summarize due to upstream error: {text}", None
        else:
             return "Error: Cannot summarize empty or invalid text.", None

    try:
        print(f"Attempting summarization using LM Studio model '{LM_STUDIO_MODEL_ID}'...")
        prompt = (
            "Analyze the following English text and provide a concise summary **in English** consisting of:\n"
            "1. Key points discussed.\n"
            "2. Action items mentioned (if any).\n\n"
            f"TEXT (Source: Whisper '{transcript_model_size}'):\n```\n{text}\n```\n\n"
            "SUMMARY (Key Points & Action Items in English):"
        )

        start_time = time.time()
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        end_time = time.time()
        print(f"Summarization finished in {end_time - start_time:.2f} seconds.")

        # Save summary to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"summaries/summary_{timestamp}.txt"
        try:
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(f"--- SOURCE TEXT (Whisper model: '{transcript_model_size}', forced English) ---\n\n")
                f.write(text)
                f.write(f"\n\n{'='*60}\n\n")
                f.write(f"--- SUMMARY (LM Studio: '{LM_STUDIO_MODEL_ID}', English) ---\n\n")
                f.write(summary)
            print(f"Summary saved to: {summary_filename}")
        except Exception as file_e:
            print(f"Error saving summary file: {file_e}")
            summary_filename = None

        return summary, summary_filename
    except openai.APIConnectionError as e:
        error_msg = f"LM Studio Connection Error: {e}. Is LM Studio running at {LM_STUDIO_BASE_URL} and the model '{LM_STUDIO_MODEL_ID}' loaded?"
        print(error_msg)
        return error_msg, None
    except Exception as e:
        error_msg = f"Error during summarization: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None

def process_upload(audio_data, selected_whisper_model_size):
    """Process uploaded or recorded audio file using the selected Whisper model"""
    if audio_data is None:
        return "No audio provided", "No summary available", None, f"No audio input received."
    if not selected_whisper_model_size:
         selected_whisper_model_size = DEFAULT_WHISPER_MODEL_SIZE
         print(f"Warning: No Whisper model size selected, defaulting to '{DEFAULT_WHISPER_MODEL_SIZE}'")

    transcript = "Error: Processing failed"
    summary = "Error: Processing failed"
    summary_file = None
    status_update = f"Initiating process with Whisper '{selected_whisper_model_size}'..."

    try:
        # Step 1: Save audio
        print("Saving audio...")
        status_update += "\n1. Saving audio..."
        audio_path = save_audio_file(audio_data)
        if not audio_path:
             fail_reason = f"Failed to save audio data (type: {type(audio_data)}, content: {str(audio_data)[:100]})"
             print(fail_reason)
             return fail_reason, "Cannot summarize", None, f"{status_update}\nError: {fail_reason}"
        print(f"Audio saved to {audio_path}")
        status_update += f" OK ({os.path.basename(audio_path)})"

        # Step 2: Transcribe
        print(f"Transcribing audio using '{selected_whisper_model_size}'...")
        status_update += f"\n2. Transcribing (Whisper '{selected_whisper_model_size}')..."
        transcript = transcribe_audio(audio_path, selected_whisper_model_size)
        print(f"Transcript generated ({selected_whisper_model_size}): {transcript[:150]}...")
        if transcript.startswith("Error:"):
             status_update += f" Error!"
             # Stop processing if transcription failed
             return transcript, "Transcription failed", None, status_update
        elif transcript == "[No speech detected]":
             status_update += f" OK (No speech detected)"
        else:
             status_update += f" OK"

        # Step 3: Summarize (only if transcription succeeded)
        print("Summarizing transcript...")
        status_update += f"\n3. Summarizing (LM Studio '{LM_STUDIO_MODEL_ID}')..."
        summary, summary_file = summarize_text(transcript, transcript_model_size=selected_whisper_model_size)
        print(f"Summary result: {summary[:150]}...")
        if summary.startswith("Error:") or summary.startswith("Cannot summarize"):
            status_update += " Error!"
        else:
            status_update += " OK"

        status_update += "\nProcessing complete."
        return transcript, summary, summary_file, status_update

    except Exception as e:
        error_msg = f"Unexpected error during audio processing: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        final_transcript = transcript if transcript and not transcript.startswith("Error:") else error_msg
        final_summary = summary if summary and not summary.startswith("Error:") else f"Processing error occurred: {error_msg}"
        status_update += f"\nSystem Error: {error_msg}"
        return final_transcript, final_summary, summary_file, status_update

def process_text(text):
    """Process direct text input"""
    if not text or not text.strip():
        return "No text provided", None, "Status: No text input."

    summary = "Error: Processing failed"
    summary_file = None
    status_update = "Initiating text summarization..."

    try:
        # --- Summarize text ---
        print("Summarizing provided text...")
        status_update += f"\n1. Summarizing (LM Studio '{LM_STUDIO_MODEL_ID}')..."
        summary, summary_file = summarize_text(text, transcript_model_size="N/A - Text Input")
        print(f"Summary generated: {summary[:150]}...")
        if summary.startswith("Error:") or summary.startswith("Cannot summarize"):
            status_update += " Error!"
        else:
             status_update += " OK"

        status_update += "\nProcessing complete."
        return summary, summary_file, status_update
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        status_update += f"\nSystem Error: {error_msg}"
        return error_msg, None, status_update

# --- Theme System ---
class ThemeManager:
    """Manages theme settings and provides CSS"""
    
    # Available themes
    THEMES = {
        "light": {
            "name": "Light",
            "primary": "#2563EB",  # Blue
            "secondary": "#4B5563",  # Gray
            "background": "#FFFFFF",  # White
            "surface": "#F9FAFB",  # Light Gray
            "text": "#111827",  # Near Black
            "border": "#E5E7EB",  # Light Gray
            "error": "#DC2626",  # Red
            "success": "#10B981",  # Green
            "accent": "#8B5CF6",  # Purple
            "font": "Inter"
        },
        "dark": {
            "name": "Dark",
            "primary": "#3B82F6",  # Blue
            "secondary": "#9CA3AF",  # Gray
            "background": "#111827",  # Very Dark Blue/Gray
            "surface": "#1F2937",  # Dark Blue/Gray
            "text": "#F9FAFB",  # Off White
            "border": "#374151",  # Medium Gray
            "error": "#EF4444",  # Red
            "success": "#10B981",  # Green
            "accent": "#8B5CF6",  # Purple
            "font": "Inter"
        },
        "midnight": {
            "name": "Midnight",
            "primary": "#818CF8",  # Indigo
            "secondary": "#94A3B8",  # Slate
            "background": "#0F172A",  # Very Dark Blue
            "surface": "#1E293B",  # Dark Blue
            "text": "#F8FAFC",  # Slate-50
            "border": "#334155",  # Slate-700
            "error": "#F87171",  # Red
            "success": "#34D399",  # Green
            "accent": "#A78BFA",  # Purple
            "font": "Inter"
        },
        "teal": {
            "name": "Teal",
            "primary": "#14B8A6",  # Teal
            "secondary": "#64748B",  # Slate
            "background": "#ECFEFF",  # Cyan-50
            "surface": "#F0FDFA",  # Teal-50
            "text": "#134E4A",  # Teal-900
            "border": "#CCFBF1",  # Teal-100
            "error": "#DC2626",  # Red
            "success": "#059669",  # Green
            "accent": "#0E7490",  # Cyan-800
            "font": "Inter"
        },
        "nord": {
            "name": "Nord",
            "primary": "#5E81AC",  # Nord Blue
            "secondary": "#81A1C1",  # Nord Light Blue
            "background": "#2E3440",  # Nord Dark
            "surface": "#3B4252",  # Nord Medium Dark
            "text": "#ECEFF4",  # Nord Snow
            "border": "#4C566A",  # Nord Darker Gray
            "error": "#BF616A",  # Nord Red
            "success": "#A3BE8C",  # Nord Green
            "accent": "#B48EAD",  # Nord Purple
            "font": "Inter"
        },
        "solarized": {
            "name": "Solarized",
            "primary": "#268BD2",  # Solarized Blue
            "secondary": "#93A1A1",  # Solarized Base1
            "background": "#FDF6E3",  # Solarized Light BG
            "surface": "#EEE8D5",  # Solarized Light Surface
            "text": "#586E75",  # Solarized Base01
            "border": "#D0D8DB",  # Custom light border
            "error": "#DC322F",  # Solarized Red
            "success": "#859900",  # Solarized Green
            "accent": "#6C71C4",  # Solarized Violet
            "font": "Inter"
        },
        "solarized-dark": {
            "name": "Solarized Dark",
            "primary": "#268BD2",  # Solarized Blue
            "secondary": "#93A1A1",  # Solarized Base1
            "background": "#002B36",  # Solarized Dark BG
            "surface": "#073642",  # Solarized Dark Surface
            "text": "#93A1A1",  # Solarized Base1
            "border": "#586E75",  # Solarized Base01
            "error": "#DC322F",  # Solarized Red
            "success": "#859900",  # Solarized Green
            "accent": "#6C71C4",  # Solarized Violet
            "font": "Inter"
        },
        "custom": {
            "name": "Custom",
            "primary": "#3B82F6",  # Blue
            "secondary": "#9CA3AF",  # Gray
            "background": "#FFFFFF",  # White
            "surface": "#F9FAFB",  # Light Gray
            "text": "#111827",  # Near Black
            "border": "#E5E7EB",  # Light Gray
            "error": "#DC2626",  # Red
            "success": "#10B981",  # Green
            "accent": "#8B5CF6",  # Purple
            "font": "Inter"
        }
    }
    
    def __init__(self, default_theme: str = "light"):
        self.current_theme = default_theme
        self.custom_theme = self.THEMES["custom"].copy()
    
    def get_theme_names(self) -> list:
        """Returns a list of available theme names"""
        return list(self.THEMES.keys())
    
    def update_custom_theme(self, **kwargs):
        """Update custom theme with provided values"""
        for key, value in kwargs.items():
            if key in self.custom_theme and value:
                self.custom_theme[key] = value
    
    def set_theme(self, theme_name: str):
        """Set the current theme"""
        if theme_name in self.THEMES:
            self.current_theme = theme_name
            return True
        return False
    
    def get_current_theme_data(self):
        """Get the current theme data"""
        if self.current_theme == "custom":
            return self.custom_theme
        return self.THEMES[self.current_theme]
    
    def generate_css(self):
        """Generate CSS based on current theme"""
        theme = self.get_current_theme_data()
        
        # Define CSS with theme variables
        css = f"""
        :root {{
            --color-primary: {theme["primary"]};
            --color-secondary: {theme["secondary"]};
            --color-background: {theme["background"]};
            --color-surface: {theme["surface"]};
            --color-text: {theme["text"]};
            --color-border: {theme["border"]};
            --color-error: {theme["error"]};
            --color-success: {theme["success"]};
            --color-accent: {theme["accent"]};
            --font-family: '{theme["font"]}', sans-serif;
        }}
        
        /* Overall app styling */
        .gradio-container {{
            font-family: var(--font-family);
            max-width: 1200px;
            margin: auto;
            padding: 1.5rem;
            background-color: var(--color-background);
            color: var(--color-text);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Header styling */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--color-text);
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        
        /* Custom header with logo styling */
        .app-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--color-border);
            position: relative;
        }}
        
        .app-title {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--color-primary);
            text-align: center;
            margin: 0;
        }}
        
        .app-subtitle {{
            text-align: center;
            color: var(--color-secondary);
            font-size: 1rem;
            margin-top: 0.25rem;
        }}
        
        /* Tabs styling */
        .tabs {{
            background-color: var(--color-surface);
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--color-border);
        }}
        
        .tab-button {{
            background-color: transparent;
            color: var(--color-text);
            border: none;
            padding: 0.75rem 1.25rem;
            font-weight: 500;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
        }}
        
        .tab-button.active {{
            background-color: var(--color-primary);
            color: white;
        }}
        
        .tab-button:hover:not(.active) {{
            background-color: rgba(0, 0, 0, 0.05);
        }}
        
        /* Card/Section styling */
        .card {{
            background-color: var(--color-surface);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--color-border);
        }}
        
        .card-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--color-text);
            display: flex;
            align-items: center;
        }}
        
        .card-title-icon {{
            margin-right: 0.5rem;
            color: var(--color-primary);
        }}
        
        /* Input controls */
        .gr-form {{
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
            background-color: var(--color-surface);
        }}
        
        /* Text inputs */
        .gr-input, .gr-textarea {{
            border: 1px solid var(--color-border);
            border-radius: 6px;
            padding: 0.75rem;
            background-color: var(--color-background);
            color: var(--color-text);
            font-family: var(--font-family);
            font-size: 1rem;
            width: 100%;
            transition: border-color 0.3s;
        }}
        
        .gr-input:focus, .gr-textarea:focus {{
            border-color: var(--color-primary);
            outline: none;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }}
        
        /* Buttons */
        .gr-button {{
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }}
        
        .gr-button-primary {{
            background-color: var(--color-primary);
            color: white;
            border: none;
        }}
        
        .gr-button-primary:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}
        
        .gr-button-secondary {{
            background-color: transparent;
            color: var(--color-primary);
            border: 1px solid var(--color-primary);
        }}
        
        .gr-button-secondary:hover {{
            background-color: rgba(59, 130, 246, 0.1);
        }}
        
        /* Result area */
        .result-box {{
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
        }}
        
        .result-title {{
            font-weight: 600;
            color: var(--color-text);
            margin-bottom: 0.5rem;
        }}
        
        .result-content {{
            white-space: pre-wrap;
            font-size: 1rem;
            line-height: 1.5;
        }}
        
        /* Status box */
        .status-box {{
            font-size: 0.9rem;
            padding: 0.75rem;
            border-radius: 6px;
            background-color: var(--color-surface);
            border: 1px dashed var(--color-border);
            color: var(--color-secondary);
            font-family: monospace;
        }}
        
        /* Theme switcher */
        .theme-switcher {{
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            display: flex;
            gap: 0.5rem;
        }}
        
        .theme-select {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            border: 1px solid var(--color-border);
            background-color: var(--color-background);
            color: var(--color-text);
            font-size: 0.8rem;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            font-size: 0.9rem;
            color: var(--color-secondary);
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--color-border);
        }}
        
        /* Hide Gradio footer */
        footer {{
            display: none !important;
        }}
        
        /* Audio component */
        .gr-audio {{
            background-color: var(--color-surface);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--color-border);
        }}
        
        /* File component */
        .gr-file {{
            background-color: var(--color-surface);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--color-border);
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .gradio-container {{
                padding: 1rem;
            }}
            
            .card {{
                padding: 1rem;
            }}
        }}
        
        /* Custom theme picker classes */
        .theme-panel {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        
        .theme-preview {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid var(--color-border);
            transition: transform 0.2s;
        }}
        
        .theme-preview:hover {{
            transform: scale(1.1);
        }}
        
        .theme-preview.active {{
            border: 2px solid var(--color-primary);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
        }}
        
        .custom-theme-panel {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 1rem;
            background-color: var(--color-surface);
            border-radius: 8px;
            border: 1px solid var(--color-border);
            margin-top: 1rem;
        }}
        
        .color-picker-group {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .color-label {{
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .color-input {{
            width: 100%;
            height: 32px;
            padding: 0.25rem;
            border-radius: 4px;
            border: 1px solid var(--color-border);
            background-color: var(--color-background);
        }}
        
        
        .processing-indicator {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 6px;
            padding: 0.75rem;
            margin-top: 1rem;
            color: var(--color-accent);
            font-size: 0.9rem;
        }}
        
        .spinner {{
            border: 2px solid rgba(0, 0, 0, 0.1);
            border-left-color: var(--color-primary);
            border-radius: 50%;
            width: 16px;
            height: 16px;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to { transform: rotate(360deg); }
        }}
        
        /* Results styling */
        .transcript-box, .summary-box {{
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
            font-size: 1rem;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
            position: relative;
        }}
        
        .copy-button {{
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        }}
        
        .copy-button:hover {{
            opacity: 1;
        }}
        """
        
        return css

# --- Main App With New Theme System ---
def create_app():
    """Create the Gradio interface with theming support"""
    # Initialize theme manager
    theme_manager = ThemeManager(default_theme="light")
    
    # Define interface components
    with gr.Blocks(title="Audio & Text Summarization", css=theme_manager.generate_css()) as app:
        # State variables
        current_theme = gr.State(value="light")
        
        # Theme selector
        def update_theme(theme_name):
            theme_manager.set_theme(theme_name)
            return theme_manager.generate_css()
        
        def update_custom_theme(primary, secondary, background, text, accent):
            theme_manager.update_custom_theme(
                primary=primary,
                secondary=secondary,
                background=background,
                text=text,
                accent=accent
            )
            theme_manager.set_theme("custom")
            return theme_manager.generate_css()
        
        # Custom header
        with gr.Row(elem_classes="app-header"):
            with gr.Column():
                gr.Markdown("# Audio & Text Summarization Tool", elem_classes="app-title")
                gr.Markdown(f"Process English audio or text using Whisper and LM Studio", elem_classes="app-subtitle")
            
            # Theme selector in header
            with gr.Column(scale=1, min_width=200):
                theme_dropdown = gr.Dropdown(
                    choices=["light", "dark", "midnight", "nord", "teal", "solarized", "solarized-dark", "custom"],
                    value="light",
                    label="Theme",
                    elem_classes="theme-select"
                )
        
        # Custom theme settings (initially hidden)
        custom_theme_panel = gr.Row(visible=False, elem_classes="custom-theme-panel")
        with custom_theme_panel:
            with gr.Column(scale=1):
                primary_color = gr.ColorPicker(value="#3B82F6", label="Primary Color")
            with gr.Column(scale=1):
                secondary_color = gr.ColorPicker(value="#9CA3AF", label="Secondary Color")
            with gr.Column(scale=1):
                background_color = gr.ColorPicker(value="#FFFFFF", label="Background")
            with gr.Column(scale=1):
                text_color = gr.ColorPicker(value="#111827", label="Text")
            with gr.Column(scale=1):
                accent_color = gr.ColorPicker(value="#8B5CF6", label="Accent")
            with gr.Column(scale=1):
                apply_btn = gr.Button("Apply Custom Theme", variant="primary")
        
        # Show/hide custom theme panel
        def toggle_custom_theme_panel(theme):
            return gr.update(visible=theme == "custom")
        
        theme_dropdown.change(toggle_custom_theme_panel, inputs=[theme_dropdown], outputs=[custom_theme_panel])
        theme_dropdown.change(update_theme, inputs=[theme_dropdown], outputs=[app])
        apply_btn.click(
            update_custom_theme,
            inputs=[primary_color, secondary_color, background_color, text_color, accent_color],
            outputs=[app]
        )
        
        # Main content with tabs
        with gr.Tabs(elem_classes="tabs"):
            # --- Process Audio Tab ---
            with gr.Tab("Process Audio", elem_classes="tab"):
                with gr.Row(elem_classes="card"):
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Input Audio", elem_classes="card-title")
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Upload or Record English Audio",
                            elem_classes="audio-input"
                        )
                        
                        gr.Markdown("### ⚙️ Settings", elem_classes="card-title")
                        whisper_model_selector = gr.Dropdown(
                            AVAILABLE_WHISPER_MODELS,
                            value=DEFAULT_WHISPER_MODEL_SIZE,
                            label="Whisper Model Size",
                            info="Larger models are more accurate but slower.",
                            elem_classes="model-select"
                        )
                        
                        process_audio_button = gr.Button(
                            "🚀 Transcribe and Summarize Audio",
                            variant="primary",
                            elem_classes="process-button"
                        )
                        
                        # Status display
                        status_output_audio = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            lines=3,
                            placeholder="Processing updates will appear here...",
                            elem_classes="status-box"
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Results", elem_classes="card-title")
                        
                        with gr.Tab("Transcript"):
                            transcript_output = gr.Textbox(
                                label="Transcript",
                                lines=12,
                                interactive=False,
                                placeholder="Transcription will appear here...",
                                elem_classes="transcript-box"
                            )
                            
                        with gr.Tab("Summary"):
                            summary_output = gr.Textbox(
                                label="Summary",
                                lines=12,
                                interactive=False,
                                placeholder="Summary will appear here...",
                                elem_classes="summary-box"
                            )
                            
                        file_output = gr.File(
                            label="Download Results",
                            file_types=[".txt"],
                            elem_classes="file-output"
                        )

                # Link components
                process_audio_button.click(
                    process_upload,
                    inputs=[audio_input, whisper_model_selector],
                    outputs=[transcript_output, summary_output, file_output, status_output_audio],
                    show_progress="full"
                )

            # --- Process Text Tab ---
            with gr.Tab("Process Text", elem_classes="tab"):
                with gr.Row(elem_classes="card"):
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Input Text", elem_classes="card-title")
                        text_input = gr.Textbox(
                            label="Paste or Type English Text Here",
                            placeholder="Enter the text you want to summarize...",
                            lines=15,
                            elem_classes="text-input"
                        )
                        
                        process_text_button = gr.Button(
                            "🚀 Summarize Text",
                            variant="primary",
                            elem_classes="process-button"
                        )
                        
                        # Status display
                        status_output_text = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            lines=2,
                            placeholder="Processing updates will appear here...",
                            elem_classes="status-box"
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Results", elem_classes="card-title")
                        
                        text_summary_output = gr.Textbox(
                            label="Summary",
                            lines=15,
                            interactive=False,
                            placeholder="Summary will appear here...",
                            elem_classes="summary-box"
                        )
                        
                        text_file_output = gr.File(
                            label="Download Results",
                            file_types=[".txt"],
                            elem_classes="file-output"
                        )

                # Link components
                process_text_button.click(
                    process_text,
                    inputs=text_input,
                    outputs=[text_summary_output, text_file_output, status_output_text],
                    show_progress="full"
                )
                
            # --- Settings Tab ---
            with gr.Tab("Settings", elem_classes="tab"):
                with gr.Row(elem_classes="card"):
                    with gr.Column():
                        gr.Markdown("### 🔧 Application Settings", elem_classes="card-title")
                        
                        gr.Markdown("#### Theme Settings")
                        with gr.Row():
                            theme_buttons = []
                            for theme_name, theme_data in theme_manager.THEMES.items():
                                if theme_name != "custom":
                                    theme_btn = gr.Button(
                                        theme_data["name"],
                                        elem_classes=f"theme-button {theme_name}"
                                    )
                                    theme_btn.click(
                                        lambda t=theme_name: update_theme(t),
                                        outputs=[app]
                                    )
                                    theme_btn.click(
                                        lambda t=theme_name: t,
                                        outputs=[theme_dropdown]
                                    )
                        
                        gr.Markdown("#### Custom Theme")
                        with gr.Row():
                            custom_primary = gr.ColorPicker(value="#3B82F6", label="Primary")
                            custom_secondary = gr.ColorPicker(value="#9CA3AF", label="Secondary")
                            custom_background = gr.ColorPicker(value="#FFFFFF", label="Background")
                        with gr.Row():
                            custom_text = gr.ColorPicker(value="#111827", label="Text")
                            custom_accent = gr.ColorPicker(value="#8B5CF6", label="Accent")
                            custom_apply = gr.Button("Apply Custom Theme", variant="primary")
                        
                        custom_apply.click(
                            update_custom_theme,
                            inputs=[
                                custom_primary, custom_secondary, custom_background, 
                                custom_text, custom_accent
                            ],
                            outputs=[app]
                        )
                        custom_apply.click(lambda: "custom", outputs=[theme_dropdown])
                        
                        gr.Markdown("#### Model Settings")
                        with gr.Row():
                            gr.Markdown(f"""
                            **Current LM Studio Model:** `{LM_STUDIO_MODEL_ID}`  
                            **LM Studio URL:** `{LM_STUDIO_BASE_URL}`  
                            **Default Whisper Model:** `{DEFAULT_WHISPER_MODEL_SIZE}`
                            """)
                
        # Footer
        with gr.Row(elem_classes="footer"):
            gr.Markdown("Audio & Text Summarization Tool | Powered by Whisper & LM Studio")
    
    return app

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing models...")
    initialize_models()

    # Check LM Studio Client
    lm_studio_ok = True
    if not client:
        print("\n" + "="*30 + " WARNING " + "="*30)
        print(f"LM Studio client failed to initialize. Summarization will NOT work.")
        print("Please check errors above. Ensure:")
        print(f"  1. LM Studio application is running.")
        print(f"  2. The LM Studio server is enabled and accessible at {LM_STUDIO_BASE_URL}.")
        print(f"  3. The correct model ('{LM_STUDIO_MODEL_ID}') is fully loaded in LM Studio.")
        print("="*69 + "\n")
        lm_studio_ok = False

    # Check Default Whisper Model Load
    whisper_default_ok = DEFAULT_WHISPER_MODEL_SIZE in loaded_whisper_models
    if not whisper_default_ok:
        print("\n" + "="*30 + " WARNING " + "="*30)
        print(f"Default Whisper model ('{DEFAULT_WHISPER_MODEL_SIZE}') failed to load during initialization.")
        print("Audio transcription might fail until a model is successfully selected and loaded.")
        print("Check previous errors for details (ffmpeg, permissions, RAM/VRAM, disk space).")
        print("="*69 + "\n")

    if not lm_studio_ok or not whisper_default_ok:
         print("!!! One or more essential components failed to initialize. The application might not function correctly. !!!")

    print("Creating Gradio interface with theme support...")
    app = create_app()

    print("Starting Gradio interface...")
    print(f"Access the interface locally at: http://127.0.0.1:7860 (or the address Gradio prints)")
    try:
         app.launch(share=False, server_name="0.0.0.0")
    except Exception as e:
         print(f"\nError launching Gradio: {e}")
         print("The default port 7860 might be in use. Try specifying a different port:")
         print("Example: app.launch(share=False, server_name='0.0.0.0', server_port=7861)")
        
        