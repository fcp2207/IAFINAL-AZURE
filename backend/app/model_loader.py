import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Modelo Hugging Face
MODEL_REPO = os.getenv("MODEL_REPO", "mistralai/Mistral-7B-Instruct-v0.1")

# Directorio de caché para ahorrar espacio
CACHE_DIR = os.getenv("HF_HOME", "/app/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global _model, _tokenizer, _device

    if _model is None or _tokenizer is None:
        print("🔄 Cargando modelo Mistral desde Hugging Face...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR)
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_REPO,
                device_map="auto",  # lo puede manejar Azure si no hay GPU
                quantization_config=quant_config,
                torch_dtype=torch.float32,
                cache_dir=CACHE_DIR
            )

            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
                _model.config.pad_token_id = _tokenizer.eos_token_id

            print(f"✅ Modelo cargado correctamente en {_device}.")
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            _model, _tokenizer = None, None

    return _model, _tokenizer, _device


if __name__ == "__main__":
    load_model()




