from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from .model_loader import load_model  # 🔥 Importamos el modelo ya cargado
from embeddings.embedding_store import EmbeddingStore  # 🔥 Importamos la búsqueda de contexto
import numpy as np

# ✅ Cargar el modelo desde el módulo sin volver a descargarlo
model, tokenizer, device = load_model()
store = EmbeddingStore()

app = FastAPI(title="Phi-2 API", description="API optimizada con GPU", version="4.2.0")

print(torch.__version__)
print(f"torch cuda aviable {torch.cuda.is_available()}")  # Debe imprimir `False`

class InputData(BaseModel):
    input_text: str

# def es_pregunta_sobre_motores(pregunta):
#     """Determina si la pregunta es sobre motores eléctricos usando palabras clave."""
#     palabras_clave = ["motor", "voltaje", "corriente", "rpm", "bobinado", "tensión", "aislamiento", "falla"]
#     return any(palabra in pregunta.lower() for palabra in palabras_clave)

def es_pregunta_sobre_motores(pregunta):
    """Determina si una pregunta es sobre motores eléctricos aplicando un umbral en FAISS."""
    
    print(f"\n🔍 Pregunta recibida: {pregunta}")

    # results = store.search(pregunta, top_k=3, umbral=0.5, max_top_k=50)  # 🔹 Usa el nuevo `search()` con umbral
    results = store.search(pregunta)

    if results:
        print(f"✅ FAISS activado. Documentos relevantes: {len(results)}")
        return results
    
    print("❌ No hay documentos relevantes. Clasificando como pregunta general.")
    return []

    # # 🔹 Verificar si FAISS está cargado
    # if store.index is None:
    #     print("⚠️ FAISS no está cargado. Intentando recargar...")
    #     store.load_index()

    # # 🔹 Generar embedding de la pregunta
    # pregunta_embedding = store.model.encode([pregunta], convert_to_numpy=True)

    # # 🔹 Incrementar dinámicamente `top_k` si no encontramos documentos relevantes
    # top_k = top_k_base
    # retrieved_docs = []

    # while top_k <= max_top_k:
    #     # 🔹 Buscar en FAISS con el valor actual de `top_k`
    #     distances, indices = store.index.search(pregunta_embedding, top_k)

    #     # 🔹 Filtrar documentos relevantes según el umbral
    #     retrieved_docs = [
    #         store.get_document(indices[0][i]) for i, dist in enumerate(distances[0])
    #         if i < len(store.docs) and dist < umbral
    #     ]

    #     print(f"📊 Intento con top_k={top_k} → Documentos encontrados: {len(retrieved_docs)}")

    #     # 🔹 Si encontramos documentos, devolvemos los resultados
    #     if retrieved_docs:
    #         print(f"✅ FAISS activado con top_k={top_k}. Documentos relevantes: {len(retrieved_docs)}")
    #         return retrieved_docs
        
    #     # 🔹 Si no encontramos nada, aumentamos `top_k` y probamos de nuevo
    #     top_k += 5

    # # 🔹 Si después de `max_top_k` intentos no hay resultados, es una pregunta general
    # print("❌ FAISS no encontró documentos relevantes. Clasificando como pregunta general.")
    # return []

    # """Determina si una pregunta es sobre motores eléctricos usando embeddings y verificación en FAISS."""

    # frases_clave = [
    #     "potencia del motor", "potencia del equipo", "potencia nominal",
    #     "corriente nominal", "prueba al vacío", "bobinado del estator",
    #     "fallas del motor", "equipo blower", "pruebas eléctricas",
    #     "tensión aplicada", "frecuencia del motor", "aislamiento eléctrico",
    #     "medición de voltaje", "análisis de vibración", "velocidad de prueba",
    #     "fases del motor", "pruebas de aislamiento", "índice de polarización",
    #     "ajuste en alojamientos", "temperatura del motor", "tipo de falla",
    #     "verificación de placa de datos", "reparación y bobinado de motores"
    # ]

    # # 🔹 Generar embeddings para frases clave y la pregunta
    # frases_embeddings = store.model.encode(frases_clave, convert_to_numpy=True)
    # pregunta_embedding = store.model.encode([pregunta], convert_to_numpy=True)

    # # 🔹 Calcular distancia con cada frase clave (menor distancia = más similar)
    # distances = np.linalg.norm(frases_embeddings - pregunta_embedding, axis=1)
    # distancia_minima = np.min(distances)

    # # 🔹 Si la pregunta es similar a frases clave, buscar en FAISS directamente
    # if distancia_minima < umbral:
    #     return store.search(pregunta, top_k=top_k_base)

    # # 🔹 Ajustar dinámicamente el `top_k` si la pregunta no pasó el primer filtro
    # top_k = top_k_base if distancia_minima < umbral + 0.05 else top_k_base * 2  

    # # 🔹 Búsqueda en FAISS para ver si la pregunta tiene relación con los documentos indexados
    # distances, indices = store.index.search(pregunta_embedding, top_k)
    # retrieved_docs = [
    #     store.get_document(indices[0][i]) for i, dist in enumerate(distances[0]) 
    #     if i < len(store.docs) and dist < umbral + 0.1
    # ]

    # # 🔹 Si FAISS encuentra información relevante, se considera pregunta sobre motores
    # if retrieved_docs:
    #     return retrieved_docs

    # # 🔹 Si no hay coincidencias, devolver None (pregunta general)
    # return []





@app.get("/")
def home():
    return {"message": "API ejecutándose 🚀"}


@app.post("/predict")
async def predict(data: InputData):
    is_motor_question = es_pregunta_sobre_motores(data.input_text)
    if is_motor_question:
        cleaned_docs = []
        print("✅ Recibida solicitud con:", data.input_text)
        # ollama_client = cl.user_session.get("ollama_client")

        # 🔎 Recuperar contexto relevante usando embeddings
        # retrieved_docs = store.search(data.input_text, top_k=3)
        retrieved_docs = is_motor_question
        # Manejar el caso en que no haya documentos relevantes
        if not retrieved_docs:
            context = "No se encontró información relevante en la base de datos."
        else:
            # Filtrar texto redundante y limitar caracteres
            for doc in retrieved_docs:
                lines = doc.split("\n")  # Dividir en líneas
                unique_lines = list(dict.fromkeys(lines))  # Eliminar duplicados
                cleaned_doc = "\n".join(unique_lines[:5])  # Solo tomar las 5 primeras líneas útiles
                cleaned_docs.append(cleaned_doc)

        # Limitar el contexto a 1000 caracteres para evitar entradas demasiado largas
        context = " ".join(cleaned_docs)[:500]  # Limitar el contexto a 500 caracteres
        prompt = f"""Eres un asistente experto en motores eléctricos con 25 años de experiencia. Usa la siguiente información para responder sin mencionar este contexto en la respuesta.

        [Contexto relevante]
        {context}

        [Instrucciones]
        Responde de forma clara y concisa basándote en el contexto relevante. No menciones que eres un asistente ni hagas referencia al contexto explícitamente.

        [Pregunta del usuario]
        {data.input_text}

        [Respuesta]"""
    else:
        # 🔹 Pregunta general → No usar embeddings
        prompt = f"""Eres un asistente general. Responde de manera clara y útil a la siguiente pregunta:

        Usuario: {data.input_text}
        Modelo:"""

    print(f"📝 Prompt generado:\n{prompt}")

    response = model.generate_response(prompt=prompt)
    

    # response = {"message": "cargue el modelo y epa"}
    response = {"message": response}
    print(response)
    # response = await cl.make_async(ollama_client.generate_response)(prompt=prompt)
    return response




    # ##breakpoint()
    # print("✅ Recibida solicitud con:", data.input_text)  # DEBUG

    # if model is None or tokenizer is None:
    #     print("❌ Error: Modelo no cargado.")
    #     raise HTTPException(status_code=500, detail="Modelo no cargado correctamente.")

    # try:
    #     # 🔎 Recuperar contexto relevante usando embeddings
    #    ## breakpoint()
    #     retrieved_docs = store.search(data.input_text, top_k=3)
    #     print("Documento recibido:", retrieved_docs)
    #     context = " ".join(retrieved_docs)

    #     # 📝 Construir nuevo prompt con el contexto
    #     prompt = f"Contexto relevante:\n{context}\n\nUsuario: {data.input_text}\nModelo:"

    #     ###inputs = tokenizer(data.input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     inputs = {key: val.to(device) for key, val in inputs.items()}  

    #     print("🔄 Generando respuesta...")
    #     # torch.cuda.empty_cache()  # Liberar memoria
    #     # torch.cuda.synchronize()  # Asegurar sincronización de procesos en GPU

    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, 
    #                                  max_new_tokens=200, 
    #                                  temperature=0.7,
    #                                  top_p=0.9,
    #                                  top_k=40,
    #                                  do_sample=True)

    #     response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print("✅ Respuesta generada:", response_text)

    #     return {"response": response_text}

    # except Exception as e:
    #     print("❌ Error en la generación:", e)
    #     raise HTTPException(status_code=500, detail=f"Error de inferencia: {str(e)}")









# import os
# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# # ✅ Configuración de variables en Hugging Face
# MODEL_REPO = os.getenv("MODEL_REPO", "fcp2207/Modelo_Phi2_fusionado")  
# CACHE_DIR = os.getenv("HF_HOME", "/app/cache")  
# FEEDBACK_FILE = os.path.join(CACHE_DIR, "feedback.json")

# # ✅ Configurar caché en Hugging Face
# os.makedirs(CACHE_DIR, exist_ok=True)
# os.environ["HF_HOME"] = CACHE_DIR
# os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR  

# # ✅ Inicializar FastAPI
# app = FastAPI(title="Phi-2 API", description="API optimizada en Hugging Face Spaces con GPU", version="4.2.0")

# # ✅ Clases para datos de entrada
# class InputData(BaseModel):
#     input_text: str

# class FeedbackData(BaseModel):
#     feedback: str  # "positivo" o "negativo"

# # ✅ Cargar feedback si existe
# def load_feedback():
#     if os.path.exists(FEEDBACK_FILE):
#         with open(FEEDBACK_FILE, "r") as f:
#             return json.load(f)
#     return {
#         "temperature": 0.6, "top_p": 0.85, "top_k": 50, 
#         "max_new_tokens": 50,  
#         "repetition_penalty": 1.2, 
#         "positivo": 0, "negativo": 0
#     }

# def save_feedback(feedback):
#     with open(FEEDBACK_FILE, "w") as f:
#         json.dump(feedback, f)

# user_feedback = load_feedback()

# # ✅ Detectar si hay GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Cargar modelo en GPU si está disponible
# try:
#     print("🔄 Cargando el modelo en Hugging Face Spaces con GPU...")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_REPO, 
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
#         device_map="auto",  
#         cache_dir=CACHE_DIR
#     )

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.eos_token_id

#     print(f"✅ Modelo cargado correctamente en {device}.")
# except Exception as e:
#     print(f"❌ Error al cargar el modelo: {str(e)}")
#     model, tokenizer = None, None

# @app.get("/")
# def home():
#     return {"message": "API ejecutándose 🚀"}

# @app.post("/predict")
# async def predict(data: InputData):
#     if model is None or tokenizer is None:
#         raise HTTPException(status_code=500, detail="Modelo no cargado correctamente.")

#     try:
#         print(data)
#         num_tokens = len(data.input_text.split())

#         # ✅ Ajustamos parámetros dinámicamente con base en feedback recibido
#         generation_params = {
#             "temperature": user_feedback["temperature"],
#             "top_p": user_feedback["top_p"],
#             "top_k": user_feedback["top_k"],
#             "max_new_tokens": user_feedback["max_new_tokens"],  
#             "do_sample": True  
#         }

#         # ✅ Corregimos la entrada para que no agregue "Responde en español:"
#         input_text = f"{data.input_text.strip()}"
#         inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

#         # ✅ Mover inputs manualmente a la GPU si está disponible
#         inputs = {key: val.to(device) for key, val in inputs.items()}

#         with torch.no_grad():
#             outputs = model.generate(**inputs, **generation_params)

#         # ✅ Eliminar la frase "Responde en español:" en caso de que siga apareciendo
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("Responde en español:", "").strip()

#         return {"response": response_text}

#     except torch.cuda.OutOfMemoryError:
#         return {"response": "⚠️ Error: Falta de memoria en GPU. Reduce la cantidad de tokens generados."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error de inferencia: {str(e)}")

# @app.post("/feedback/")
# async def receive_feedback(feedback: FeedbackData):
#     """Endpoint para recibir feedback y ajustar la generación de texto."""
#     global user_feedback

#     if feedback.feedback == "positivo":
#         user_feedback["positivo"] += 1
#         user_feedback["temperature"] = max(0.3, user_feedback["temperature"] - 0.05)  
#         user_feedback["top_p"] = min(1.0, user_feedback["top_p"] + 0.05)  
#     elif feedback.feedback == "negativo":
#         user_feedback["negativo"] += 1
#         user_feedback["temperature"] = min(1.0, user_feedback["temperature"] + 0.05)  
#         user_feedback["top_p"] = max(0.7, user_feedback["top_p"] - 0.05)  
#         user_feedback["max_new_tokens"] = max(30, user_feedback["max_new_tokens"] - 10)  

#     save_feedback(user_feedback)
    
#     return {"message": f"Feedback {feedback.feedback} recibido y parámetros ajustados"}