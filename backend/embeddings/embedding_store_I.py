import faiss
import numpy as np
import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Configuración del modelo de embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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
        self.index = faiss.IndexFlatL2(384)  # 384 = tamaño del embedding
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

            # 🔹 Filtrar documentos relevantes
            documentos_relevantes = [
                self.docs[i] for i, dist in enumerate(distances[0]) if dist <= umbral
            ]

            print(f"🔍 Intento con top_k={top_k}, umbral={umbral:.2f} → Distancias FAISS: {distances[0]}")

            if documentos_relevantes:
                print(f"✅ FAISS activado con top_k={top_k}. Documentos relevantes: {len(documentos_relevantes)}")

                respuestas = [self.extraer_respuesta_mas_relevante(doc, query) for doc in documentos_relevantes]
                respuestas_validas = [res for res in respuestas if "❌ No encontré información exacta" not in res]

                return respuestas_validas if respuestas_validas else ["⚠️ No encontré información precisa en los documentos."]

            top_k += 5  # 🔥 Aumentar `top_k` si no encuentra documentos

        print("❌ FAISS no encontró documentos relevantes. Clasificando como pregunta general.")
        return []
    # def search(self, query, top_k=5, umbral_inicial=0.8, max_top_k=50):
        # """Busca en FAISS y extrae la información más relevante de los documentos recuperados."""
        # query_embedding = self.model.encode([query], convert_to_numpy=True)
        # umbral = umbral_inicial  

        # while top_k <= max_top_k:
        #     distances, indices = self.index.search(query_embedding, top_k)

        #     # 🔹 Ajuste del umbral basado en la mejor coincidencia
        #     distancia_minima = min(distances[0]) if len(distances[0]) > 0 else float('inf')
        #     if distancia_minima > umbral and distancia_minima < 1.1:
        #         umbral = distancia_minima + 0.05  

        #     # 🔹 Filtrar documentos relevantes
        #     documentos_relevantes = [
        #         self.docs[i] for i, dist in enumerate(distances[0]) 
        #         if dist <= umbral
        #     ]

        #     print(f"🔍 Intento con top_k={top_k}, umbral={umbral:.2f} → Distancias FAISS: {distances[0]}")

        #     if documentos_relevantes:
        #         print(f"✅ FAISS activado con top_k={top_k}, umbral={umbral:.2f}. Documentos relevantes: {len(documentos_relevantes)}")

        #         # 🔥 Buscar el fragmento más relevante usando **similitud de coseno**
        #         respuestas = [self.extraer_respuesta_mas_relevante(doc, query) for doc in documentos_relevantes]

        #         # 🔹 Filtrar respuestas realmente útiles
        #         respuestas_validas = [res for res in respuestas if "❌ No encontré información exacta" not in res]

        #         return respuestas_validas if respuestas_validas else ["⚠️ No encontré información precisa en los documentos."]

        #     top_k += 2  # 🔥 Aumenta `top_k` dinámicamente si no hay documentos relevantes.

        # print("❌ FAISS no encontró documentos relevantes. Clasificando como pregunta general.")
        # return []

    def extraer_respuesta_mas_relevante(self, documento, query):
        """Encuentra el fragmento más relevante dentro de un documento basado en la consulta usando similitud de coseno."""
        palabras_clave = query.lower().split()  
        fragmentos = documento.split("\n")  

        # 🔹 Obtener embeddings de cada fragmento
        fragment_embeddings = np.array([
            self.model.encode([frag], convert_to_numpy=True)[0] for frag in fragmentos
        ])

        # 🔹 Calcular similitud de coseno entre la pregunta y los fragmentos
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        similitudes = cosine_similarity([query_embedding], fragment_embeddings)[0]

        # 🔹 Ordenar fragmentos según la similitud
        fragmentos_ordenados = sorted(
            zip(fragmentos, similitudes), key=lambda x: x[1], reverse=True
        )

        print("\n🔎 Análisis de fragmentos:")
        for frag, sim in fragmentos_ordenados:
            print(f"  - 📌 {frag} (Similitud: {sim:.2f})")

        mejor_fragmento, mejor_similitud = fragmentos_ordenados[0]

        if mejor_similitud > 0.6:  
            print(f"🎯 Fragmento ALTAMENTE relevante encontrado: {mejor_fragmento} (Similitud: {mejor_similitud:.2f})")
            return mejor_fragmento
        elif mejor_similitud > 0.4:  
            print(f"🔎 Fragmento PARCIALMENTE relevante encontrado: {mejor_fragmento} (Similitud: {mejor_similitud:.2f})")
            return mejor_fragmento
        else:
            return "❌ No encontré información exacta para tu pregunta."

    


    # def search(self, query, top_k=3):
    #     """Busca los documentos más relevantes para una consulta."""
    #     query_embedding = self.model.encode([query], convert_to_numpy=True)
    #     distances, indices = self.index.search(query_embedding, top_k)
    #     return [self.docs[i] for i in indices[0] if i < len(self.docs)]

    def save_index(self):
        """Guarda el índice y los documentos en disco."""
        faiss.write_index(self.index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(self.docs, f)

    def load_index(self):
        """Carga el índice y los documentos desde el disco."""
        self.index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, "rb") as f:
            self.docs = pickle.load(f)

    @staticmethod
    def preprocess_text(text):
        """Preprocess the text to improve embedding quality."""
        # Lowercase the text
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()

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

        # 🔹 Crear fragmentos más pequeños (máximo 2 líneas)
        fragmentos = ["\n".join(fragmentos[i:i+2]) for i in range(0, len(fragmentos), 2)]

        return list(dict.fromkeys(fragmentos))


    def load_json_data(self):
        """Carga los datos desde el JSON, extrae información útil y la indexa en FAISS."""
        if not os.path.exists(DATA_FILE):
            print(f"❌ No se encontró {DATA_FILE}. Asegúrate de que el archivo existe.")
            return
        
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        fragmentos = []

        if isinstance(data, dict):
            fragmentos = EmbeddingStore.extraer_texto_desde_json(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    fragmentos.extend(EmbeddingStore.extraer_texto_desde_json(item))

        if not fragmentos:
            print("⚠️ No se encontraron fragmentos adecuados en el JSON.")
            return

        print(f"📄 Se extrajeron {len(fragmentos)} fragmentos de texto.")

        self.add_documents(fragmentos)

        print("✅ Fragmentos añadidos al índice FAISS.")

# 🔥 Ejecutar solo si se llama directamente
if __name__ == "__main__":
    store = EmbeddingStore()
    query = "potencia del equipo"
    results = store.search(query, top_k=30, umbral=0.65)  # Reducimos el umbral
    print("\n🔎 Nueva prueba con umbral 0.65 y top_k=30:")
    print(results)


    # store.load_json_data()
    # print(f"\n📌 FAISS tiene {store.index.ntotal} documentos indexados.")
    # print("🔎 Prueba de búsqueda:")
    # query = "¿Cuál es la potencia del equipo BLOWER?"  # 🔹 Cambia esto según lo que necesites probar
    # results = store.search(query, top_k=20, umbral=0.95, max_top_k=30)
    # if results:
    #     print("\n✅ Respuesta encontrada:")
    #     for res in results:
    #         print(f" - {res}")
    # else:
    #     print("\n❌ FAISS sigue sin encontrar documentos.")
    # print("📌 Verificando si 'potencia' está en los documentos indexados...\n")

    # for i, fragment in enumerate(store.docs):
    #     if "potencia" in fragment.lower():
    #         print(f"✅ Fragmento {i} contiene 'potencia': {fragment[:200]}...")


    # store = EmbeddingStore()
    # store.load_json_data()

    # pregunta = "¿Cuál es la potencia del equipo BLOWER?"
    # query_embedding = store.model.encode([pregunta], convert_to_numpy=True)

    # # 🔹 Ver si FAISS realmente está buscando
    # print("\n🔍 Probando búsqueda manual en FAISS...")
    # distances, indices = store.index.search(query_embedding, 10)
    # print(f"📊 Distancias obtenidas: {distances[0]}")

    # # 🔹 Ver los documentos recuperados
    # retrieved_docs = [
    #     store.get_document(indices[0][i]) for i, dist in enumerate(distances[0]) 
    #     if i < len(store.docs) and dist < 0.5  # Probamos con umbral más alto
    # ]

    # if retrieved_docs:
    #     print(f"✅ FAISS encontró {len(retrieved_docs)} documentos relevantes:")
    #     for doc in retrieved_docs:
    #         print(f" - {doc[:200]}...")  # Mostrar los primeros 200 caracteres
    # else:
    #     print("❌ FAISS no encontró documentos. Puede que el índice esté vacío o el umbral sea muy estricto.")
