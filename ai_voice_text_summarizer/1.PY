import os
import time
import gradio as gr
# CHANGE: Import the Monochrome theme
from gradio.themes.monochrome import Monochrome
import openai
import whisper
from datetime import datetime
import shutil
import scipy.io.wavfile as wavfile
import numpy as np
import traceback # Import traceback for detailed error logging

# --- Create Necessary Directories ---
os.makedirs("audio", exist_ok=True)
os.makedirs("summaries", exist_ok=True)
os.makedirs("models", exist_ok=True) # Create models directory

# --- LM Studio Configuration ---
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY = "lm-studio" # Not needed for LM Studio but good practice
LM_STUDIO_MODEL_ID = "deepseek-r1-distill-qwen-7b" # Your model ID

# --- Whisper Configuration ---
WHISPER_MODEL_PATH = "./models" # Directory to store/load Whisper models
AVAILABLE_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
DEFAULT_WHISPER_MODEL_SIZE = "base"

# --- Global variables ---
loaded_whisper_models = {}
client = None

# --- Model Loading ---
# (load_whisper_model_by_size function remains the same)
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
        # Use download_root to specify the directory
        model = whisper.load_model(size, download_root=WHISPER_MODEL_PATH)
        end_time = time.time()
        print(f"Whisper model '{size}' loaded successfully in {end_time - start_time:.2f} seconds.")
        loaded_whisper_models[size] = model # Cache the loaded model
        return model, None
    except Exception as e:
        error_msg = f"Error loading Whisper model '{size}': {e}"
        print(error_msg)
        print("Please ensure 'openai-whisper' is installed, 'ffmpeg' is available,")
        print(f"the model can be downloaded to '{WHISPER_MODEL_PATH}', and you have enough RAM/VRAM.")
        # traceback.print_exc() # Uncomment for very detailed debug logs
        return None, error_msg

# (initialize_models function remains the same)
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
# (save_audio_file function remains the same)
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

# (transcribe_audio function remains the same)
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
        traceback.print_exc() # Log detailed error
        return error_msg

# (summarize_text function remains the same)
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
            f"TEXT (Source: Whisper '{transcript_model_size}'):\n```\n{text}\n```\n\n" # Indicate source model
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
                f.write(f"--- SOURCE TEXT (Whisper model: '{transcript_model_size}', forced English) ---\n\n") # Use actual model size
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

# (process_upload function remains the same)
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

# (process_text function remains the same)
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

# --- Gradio UI ---

def create_app():
    """Create the Gradio interface"""
    # CHANGE: Define custom CSS for monochrome theme refinement
    custom_css = """
    /* Overall container */
    .gradio-container {
        font-family: 'Inter', sans-serif;
        max-width: 1000px; /* Slightly wider */
        margin: auto;
        padding-top: 1rem; /* Add some top padding */
    }

    /* Hide Gradio footer */
    footer { display: none !important; }

    /* General component spacing */
    .gr-block { margin-bottom: 1rem; } /* Add space below blocks */

    /* Textbox styling */
    .gr-textbox textarea {
        font-size: 1rem;
        line-height: 1.5; /* Improve readability */
        border-radius: 6px; /* Slightly rounder corners */
        border-color: #E5E7EB; /* Lighter border for monochrome */
    }
    /* Style placeholder text */
    .gr-textbox textarea::placeholder {
        color: #9CA3AF; /* Lighter placeholder */
    }

    /* Button styling */
    .gr-button {
        margin-top: 0.5rem; /* Adjust top margin */
        border-radius: 6px;
        font-weight: 500; /* Medium weight */
    }
    /* Primary button styling */
    .gr-button.gr-button-primary {
        /* Rely on theme's primary color, maybe adjust if needed */
        /* background-color: #374151; /* Darker Gray */
        /* border-color: #374151; */
    }

    /* Status box styling */
    .status-box .gr-textbox textarea { /* Target textarea inside status box */
        font-size: 0.9em;
        color: #4B5563; /* Dark Gray text */
        background-color: #F3F4F6; /* Very light gray background */
        border: 1px dashed #D1D5DB; /* Dashed border */
        padding: 8px;
        line-height: 1.4;
    }

    /* Audio component adjustments (optional) */
    .gr-audio { /* Target the audio component block */
       /* Add specific styling if needed */
    }

    /* File output component */
    .gr-file {
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .gr-file .gr-button { /* Style button inside file component */
        font-size: 0.9em;
    }

    /* Tab styling */
    .gr-tabs button { /* Style tab buttons */
         font-weight: 500;
         padding: 0.75rem 1rem;
    }

    /* Headings */
    h1 { /* Main title */
        text-align: center;
        color: #111827; /* Very dark gray / near black */
        margin-bottom: 0.5rem;
    }
    .gr-markdown p { /* Paragraphs in markdown */
        color: #4B5563; /* Dark Gray */
        text-align: center;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    /* Component Labels */
    label span { /* Target the actual label text */
        font-weight: 500 !important; /* Ensure labels are visible */
        color: #374151 !important; /* Dark gray labels */
    }
    """
    # CHANGE: Apply Monochrome theme and custom CSS
    with gr.Blocks(title="Audio & Text Summarization",
                   theme=Monochrome(
                       primary_hue="slate", # Using slate for monochrome tones
                       secondary_hue="gray",
                       neutral_hue="gray",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
                   ),
                   css=custom_css) as app:

        gr.Markdown("# Audio & Text Summarization Tool")
        gr.Markdown(f"Process English audio or text. Uses Whisper for transcription and LM Studio (`{LM_STUDIO_MODEL_ID}`) for summarization.")

        with gr.Tabs():
            # --- Process Audio Tab ---
            with gr.Tab("Process Audio"):
                gr.Markdown("### 1. Provide Audio & Select Model") # Add subheading
                with gr.Row(equal_height=False): # Allow columns to have different heights
                    with gr.Column(scale=3): # Give more space to audio input
                        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record English Audio")
                    with gr.Column(scale=2): # Give slightly less space to dropdown
                        whisper_model_selector = gr.Dropdown(
                            AVAILABLE_WHISPER_MODELS,
                            value=DEFAULT_WHISPER_MODEL_SIZE,
                            label="Whisper Model Size", # Shorter label
                            info="Larger models are more accurate but slower."
                        )
                gr.Markdown("### 2. Process") # Add subheading
                process_audio_button = gr.Button("Transcribe and Summarize Audio", variant="primary", icon="▶️") # Add icon
                gr.Markdown("### 3. Results") # Add subheading
                with gr.Row():
                    transcript_output = gr.Textbox(label="Transcript", lines=12, interactive=False, placeholder="Transcription will appear here...")
                    summary_output = gr.Textbox(label="Summary", lines=12, interactive=False, placeholder="Summary (Key Points & Action Items) will appear here...")
                with gr.Row():
                     with gr.Column(scale=1):
                           file_output = gr.File(label="Download Results (.txt)")
                     with gr.Column(scale=2):
                           status_output_audio = gr.Textbox(label="Status", interactive=False, lines=3, placeholder="Processing updates...", elem_classes=["status-box"]) # Add class for CSS


                # Link components
                process_audio_button.click(
                    process_upload,
                    inputs=[audio_input, whisper_model_selector],
                    outputs=[transcript_output, summary_output, file_output, status_output_audio],
                    show_progress="full"
                )

            # --- Process Text Tab ---
            with gr.Tab("Process Text"):
                gr.Markdown("### 1. Enter Text") # Add subheading
                text_input = gr.Textbox(label="Paste or Type English Text Here", placeholder="Enter the text you want to summarize...", lines=15)
                gr.Markdown("### 2. Process") # Add subheading
                process_text_button = gr.Button("Summarize Text", variant="primary", icon="▶️") # Add icon
                gr.Markdown("### 3. Results") # Add subheading
                text_summary_output = gr.Textbox(label="Summary", lines=12, interactive=False, placeholder="Summary (Key Points & Action Items) will appear here...")
                with gr.Row():
                     with gr.Column(scale=1):
                           text_file_output = gr.File(label="Download Results (.txt)")
                     with gr.Column(scale=2):
                           status_output_text = gr.Textbox(label="Status", interactive=False, lines=2, placeholder="Processing updates...", elem_classes=["status-box"]) # Add class for CSS


                # Link components
                process_text_button.click(
                    process_text,
                    inputs=text_input,
                    outputs=[text_summary_output, text_file_output, status_output_text],
                    show_progress="full"
                )
    return app

# --- Main Execution ---
# (Main execution block remains the same)
if __name__ == "__main__":
    print("Initializing models...")
    initialize_models() # Loads default Whisper model and LM Studio client

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

    print("Creating Gradio interface...")
    app = create_app()

    print("Starting Gradio interface...")
    print(f"Access the interface locally at: http://127.0.0.1:7860 (or the address Gradio prints)")
    try:
         app.launch(share=False, server_name="0.0.0.0") # Accessible on local network
    except Exception as e:
         print(f"\nError launching Gradio: {e}")
         print("The default port 7860 might be in use. Try specifying a different port:")
         print("Example: app.launch(share=False, server_name='0.0.0.0', server_port=7861)")