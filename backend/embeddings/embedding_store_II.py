import faiss
import numpy as np
import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.pairwise import cosine_similarity

# Configuración del modelo de embeddings
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Modelo actualizado
INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "docs.pkl"
DATA_FILE = "data.json"  # Archivo JSON con los datos

class EmbeddingStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.docs = []
        
        if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
            self.load_index()
        else:
            self.create_index()

    def create_index(self):
        """Crea un índice FAISS vacío."""
        self.index = faiss.IndexFlatL2(768)  # all-mpnet-base-v2 usa 768 dimensiones
        self.docs = []

    def add_documents(self, texts):
        """Convierte textos en embeddings y los agrega al índice FAISS."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        self.docs.extend(texts)
        self.save_index() 
      
    def search(self, query, top_k=10, umbral=1.01, max_top_k=30):
        """Busca en FAISS con un `top_k` dinámico para mejorar la recuperación de información."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        while top_k <= max_top_k:
            distances, indices = self.index.search(query_embedding, top_k)

            documentos_relevantes = [
                self.docs[i] for i, dist in enumerate(distances[0]) if dist <= umbral
            ]

            print(f"🔍 Intento con top_k={top_k}, umbral={umbral:.2f} → Distancias FAISS: {distances[0]}")

            if documentos_relevantes:
                print(f"✅ FAISS activado con top_k={top_k}. Documentos relevantes: {len(documentos_relevantes)}")
                return documentos_relevantes

            top_k += 5  # Aumentar `top_k` si no encuentra documentos

        print("❌ FAISS no encontró documentos relevantes.")
        return []

    def save_index(self):
        """Guarda el índice y los documentos en disco."""
        faiss.write_index(self.index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(self.docs, f)

    def load_index(self):
        """Carga el índice y los documentos desde el disco."""
        if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(DOCS_FILE, "rb") as f:
                self.docs = pickle.load(f)
        else:
            print("❌ No se encontraron archivos de índice previos. Creando uno nuevo.")
            self.create_index()

    def load_json_data(self):
        """Carga los datos desde el JSON, extrae información útil y la indexa en FAISS."""
        if not os.path.exists(DATA_FILE):
            print(f"❌ No se encontró {DATA_FILE}. Asegúrate de que el archivo existe.")
            return
        
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        fragmentos = self.extraer_texto_desde_json(data)
        if not fragmentos:
            print("⚠️ No se encontraron fragmentos adecuados en el JSON.")
            return

        print(f"📄 Se extrajeron {len(fragmentos)} fragmentos de texto.")
        self.add_documents(fragmentos)
        print("✅ Fragmentos añadidos al índice FAISS.")

    @staticmethod
    def extraer_texto_desde_json(data):
        """Extrae fragmentos de texto asegurando que cada entrada tenga contexto claro."""
        fragmentos = []

        def recorrer_json(obj, contexto=""):
            if isinstance(obj, dict):
                for clave, valor in obj.items():
                    nuevo_contexto = f"{contexto} > {clave}".strip() if contexto else clave
                    if isinstance(valor, str) and len(valor) > 2:
                        fragmentos.append(f"{nuevo_contexto}: {valor}")
                    elif isinstance(valor, list):
                        for item in valor:
                            if isinstance(item, str):
                                fragmentos.append(f"{nuevo_contexto}: {item}")
                            elif isinstance(item, dict):
                                recorrer_json(item, nuevo_contexto)
                    elif isinstance(valor, dict):
                        recorrer_json(valor, nuevo_contexto)

        recorrer_json(data)
        return list(dict.fromkeys(fragmentos))  # Eliminar duplicados

# 🔥 Ejecutar solo si se llama directamente
if __name__ == "__main__":
    store = EmbeddingStore()
    # store.load_json_data()  # Cargar datos desde JSON
    print(f"\n📌 FAISS tiene {store.index.ntotal} documentos indexados.")
    print("🔎 Prueba de búsqueda:")
    # query = "potencia del equipo"
    # Prueba con términos más generales
    results = store.search("pruebas eléctricas aislamiento resistencia", top_k=50, umbral=1.10)

    # results = store.search("potencia del equipo", top_k=30, umbral=1.05)
    print("\n🔎 Resultados mejorados:")
    print(results)
    










    # queries = [
    # "tipo de falla",
    # "servicio realizado en el equipo",
    # "pruebas eléctricas realizadas",
    # "empresa responsable de la reparación"
    # ]
    # for query in queries:
    #     print(f"\n🔍 Buscando: {query}")
    #     print(store.search(query, top_k=20, umbral=1.10))

    # for i, doc in enumerate(store.docs):
    #     if "potencia" in doc.lower():
    #         print(f"📄 Documento {i}: {doc}")


   # Ver los primeros 5 embeddings almacenados en FAISS
    # if store.index.ntotal > 0:
    #     query_embedding = store.model.encode(["potencia del equipo"], convert_to_numpy=True, normalize_embeddings=True)
        
    #     distances, indices = store.index.search(query_embedding, 5)
    #     print(f"🔍 Distancias encontradas: {distances}")
    #     print(f"🔍 Índices encontrados: {indices}")

    #     # Ver si los índices corresponden a documentos
    #     for i in indices[0]:
    #         if i < len(store.docs):
    #             print(f"📄 Documento encontrado: {store.docs[i]}")
    #         else:
    #             print(f"⚠️ Índice fuera de rango: {i}")

    # else:
    #     print("❌ No hay documentos en FAISS.")




    # # # store.load_json_data()  # Cargar datos desde JSON
    # print(f"\n📌 FAISS tiene {store.index.ntotal} documentos indexados.")
    # print("🔎 Prueba de búsqueda:")
    # query = "potencia del equipo"
    # results = store.search("potencia del equipo", top_k=10, umbral=0.90)
    # print("\n🔎 Resultados mejorados:")
    # print(results)

