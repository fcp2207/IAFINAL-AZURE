# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from .ollama_client import OllamaClient

# ### ✅ Configuración del modelo
# MODEL_REPO = os.getenv("MODEL_REPO", "fcp2207/Modelo_Phi2_fusionado")  
# CACHE_DIR = os.getenv("HF_HOME", "/app/cache")  
# os.makedirs(CACHE_DIR, exist_ok=True)

# ✅ Definir variable global para almacenar el modelo y tokenizer
_model = None
_tokenizer = None
_device = None

def load_model():
    """Carga el modelo una sola vez y lo reutiliza"""
    global _model, _tokenizer, _device

    system_instruction = ("Eres un experto en motores electricos con 25 años de experiencia y todas tus respuestas son en español")

    _model = OllamaClient(model_name='mistral:latest',
                          system_instruction=system_instruction)


    

    # if _model is None or _tokenizer is None:  # 🚀 Evita recargar si ya está en memoria
    #     print("🔄 Cargando el modelo desde Hugging Face...")
    #     try:
    #         _model = AutoModelForCausalLM.from_pretrained(
    #             MODEL_REPO,
    #             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16, #torch.float32,  
    #             device_map="auto" if torch.cuda.is_available() else None,  
    #             cache_dir=CACHE_DIR
    #         )

    #         _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR)

    #         if _tokenizer.pad_token is None:
    #             _tokenizer.pad_token = _tokenizer.eos_token
    #             _model.config.pad_token_id = _tokenizer.eos_token_id

    #         torch.compile(_model)
    #         _model.to(_device)  # 🔄 Mover a CPU/GPU según configuración

    #         print(f"✅ Modelo cargado en {_device}.")
    #     except Exception as e:
    #         print(f"❌ Error al cargar el modelo: {str(e)}")
    #         _model, _tokenizer = None, None

    return _model, _tokenizer, _device


if __name__ == "__main__":
    #  ollama_client = OllamaClient(model_name='mistral:latest')
    list_models = OllamaClient.list_models()
    print(list_models)

########### IMPORT HUGGING FACE
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # ✅ Configuración del modelo
# MODEL_REPO = os.getenv("MODEL_REPO", "fcp2207/Modelo_Phi2_fusionado")  
# CACHE_DIR = os.getenv("HF_HOME", "/app/cache")  
# os.makedirs(CACHE_DIR, exist_ok=True)

# # ✅ Definir variable global para almacenar el modelo y tokenizer
# _model = None
# _tokenizer = None
# _device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_model():
#     """Carga el modelo una sola vez y lo reutiliza"""
#     global _model, _tokenizer, _device

#     if _model is None or _tokenizer is None:  # 🚀 Evita recargar si ya está en memoria
#         print("🔄 Cargando el modelo desde Hugging Face...")
#         try:
#             _model = AutoModelForCausalLM.from_pretrained(
#                 MODEL_REPO,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16, #torch.float32,  
#                 device_map="auto" if torch.cuda.is_available() else None,  
#                 cache_dir=CACHE_DIR
#             )

#             _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR)

#             if _tokenizer.pad_token is None:
#                 _tokenizer.pad_token = _tokenizer.eos_token
#                 _model.config.pad_token_id = _tokenizer.eos_token_id

#             torch.compile(_model)
#             _model.to(_device)  # 🔄 Mover a CPU/GPU según configuración

#             print(f"✅ Modelo cargado en {_device}.")
#         except Exception as e:
#             print(f"❌ Error al cargar el modelo: {str(e)}")
#             _model, _tokenizer = None, None

#     return _model, _tokenizer, _device

######## NO SIRVE PA NADA
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # ✅ Configuración
# MODEL_REPO = os.getenv("MODEL_REPO", "fcp2207/Modelo_Phi2_fusionado")  
# CACHE_DIR = os.getenv("HF_HOME", "/app/cache")  
# os.makedirs(CACHE_DIR, exist_ok=True)

# # ✅ Detectar GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Cargar el modelo solo una vez
# print("🔄 Cargando el modelo desde Hugging Face...")
# try:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_REPO, 
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
#         device_map="auto" if torch.cuda.is_available() else None,  
#         cache_dir=CACHE_DIR
#     )

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.eos_token_id

#     print(f"✅ Modelo cargado en {device}.")
# except Exception as e:
#     print(f"❌ Error al cargar el modelo: {str(e)}")
#     model, tokenizer = None, None

# # ✅ Exportamos el modelo y tokenizer
# def get_model():
#     return model, tokenizer, device




