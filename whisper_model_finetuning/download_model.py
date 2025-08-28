import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer

# Define model names
original_model_name = "Tejveer12/whisper-finetuned"
local_model_name = "whisper-finetuned"  # Local folder name

print("Downloading and saving model locally...")

# Load model in half-precision to reduce RAM usage
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    original_model_name,
    torch_dtype=torch.float16  # Use fp16 to save memory
)

# Save model locally with sharding to avoid MemoryError
model.save_pretrained(f"./{local_model_name}", safe_serialization=False, max_shard_size="2GB")

# Save processor and tokenizer
processor = AutoProcessor.from_pretrained(original_model_name)
processor.save_pretrained(f"./{local_model_name}")

tokenizer = WhisperTokenizer.from_pretrained(original_model_name)
tokenizer.save_pretrained(f"./{local_model_name}")

print(f"✅ Model saved locally in: {local_model_name}")
