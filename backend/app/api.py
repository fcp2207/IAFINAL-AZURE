from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

from embeddings.embedding_store import EmbeddingStore  # 🔥 Importamos la búsqueda de contexto

# 🔧 Variables globales
model, tokenizer, device = None, None, None
store = EmbeddingStore()

app = FastAPI(title="Phi-2 API", description="API optimizada con GPU", version="4.2.0")

print(torch.__version__)
print(f"torch cuda available {torch.cuda.is_available()}")  # Debe imprimir `False` en Azure

# 🧪 Modo de prueba para evitar errores al levantar el contenedor
class MockModel:
    def generate_response(self, prompt):
        return "⚠️ Modo simulación. El modelo real no fue cargado."

model = MockModel()


# 🔄 Cargar modelo al iniciar
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    try:
        from .model_loader import load_model
        model, tokenizer, device = load_model()
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando el modelo real: {e}")
        model = MockModel()


class InputData(BaseModel):
    input_text: str


def es_pregunta_sobre_motores(pregunta):
    print(f"\n🔍 Pregunta recibida: {pregunta}")
    results = store.search(pregunta)

    if results:
        print(f"✅ FAISS activado. Documentos relevantes: {len(results)}")
        return results

    print("❌ No hay documentos relevantes. Clasificando como pregunta general.")
    return []


@app.get("/")
def home():
    return {"message": "API ejecutándose 🚀"}


@app.post("/predict")
async def predict(data: InputData):
    is_motor_question = es_pregunta_sobre_motores(data.input_text)

    if is_motor_question:
        cleaned_docs = []
        retrieved_docs = is_motor_question
        if not retrieved_docs:
            context = "No se encontró información relevante en la base de datos."
        else:
            for doc in retrieved_docs:
                lines = doc.split("\n")
                unique_lines = list(dict.fromkeys(lines))
                cleaned_doc = "\n".join(unique_lines[:5])
                cleaned_docs.append(cleaned_doc)

        context = " ".join(cleaned_docs)[:500]
        prompt = f"""Eres un asistente experto en motores eléctricos con 25 años de experiencia. Usa la siguiente información para responder sin mencionar este contexto en la respuesta.

        [Contexto relevante]
        {context}

        [Instrucciones]
        Responde de forma clara y concisa basándote en el contexto relevante. No menciones que eres un asistente ni hagas referencia al contexto explícitamente.

        [Pregunta del usuario]
        {data.input_text}

        [Respuesta]"""
    else:
        prompt = f"""Eres un asistente general. Responde de manera clara y útil a la siguiente pregunta:

        Usuario: {data.input_text}
        Modelo:"""

    print(f"📝 Prompt generado:\n{prompt}")

    try:
        response_text = model.generate_response(prompt=prompt)
        response = {"message": response_text}
    except Exception as e:
        response = {"message": f"❌ Error al generar respuesta: {str(e)}"}

    print(response)
    return response
