import faiss
import numpy as np
import json
import os
import pickle
from sentence_transformers import SentenceTransformer

# Configuración del modelo de embeddings
EMBEDDING_MODEL = "all-mpnet-base-v2"
INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "docs.pkl"
DATA_FILE = "data.json"

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
        """Crea un índice FAISS vacío optimizado con HNSW."""
        self.index = faiss.IndexHNSWFlat(768, 32)
        self.docs = []

    def add_documents(self, texts):
        """Convierte textos en embeddings y los agrega al índice FAISS."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embeddings)
        self.docs.extend(texts)
        self.save_index()
    
    def load_json_data(self):
        """Carga los datos desde el JSON y los indexa en FAISS con mejoras."""
        if not os.path.exists(DATA_FILE):
            print(f"❌ No se encontró {DATA_FILE}. Asegúrate de que el archivo existe.")
            return
        
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        fragmentos = self.procesar_json(data)
        if not fragmentos:
            print("⚠️ No se encontraron fragmentos adecuados en el JSON.")
            return

        print(f"📄 Se extrajeron {len(fragmentos)} fragmentos de texto.")
        self.add_documents(fragmentos)
        print("✅ Fragmentos añadidos al índice FAISS.")

    @staticmethod
    def procesar_json(data):
        """Extrae y optimiza fragmentos de texto para la indexación."""
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
        fragmentos = list(set(fragmentos))  # Eliminar duplicados
        fragmentos = [f.replace("_", " ") for f in fragmentos]  # Mejor formato de texto
        return fragmentos
    
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

    def search(self, query, top_k=10, umbral=1.0):
        """Busca los documentos más relevantes en FAISS dado un query."""
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ El índice FAISS está vacío. Asegúrate de cargar documentos.")
            return []

        # Generar embedding de la consulta con normalización
        query_embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        # Buscar en FAISS
        distances, indices = self.index.search(np.array([query_embedding]), top_k)

        # Filtrar por umbral
        resultados = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1 and dist <= umbral:
                resultados.append(self.docs[i])

        print(f"✅ FAISS activado con top_k={top_k}, umbral={umbral}. Documentos relevantes: {len(resultados)}\n")
        return resultados


# 🔥 Ejecutar si se llama directamente
# 🔥 Ejecutar solo si se llama directamente
if __name__ == "__main__":
    store = EmbeddingStore()
    # store.load_json_data()  # Cargar datos desde JSON
    print(f"\n📌 FAISS tiene {store.index.ntotal} documentos indexados.")
    print("🔎 Prueba de búsqueda:")
    # Definir diferentes queries para evaluar
    queries = [
        "equipo BLOWER?",
        "potencia del equipo",
        "fallas reportadas",
        "pruebas eléctricas aislamiento resistencia",
        "tipo de falla sobrecarga recomendaciones repuestos"
    ]

    # Rango de valores para top_k y umbral
    top_k_values = [10, 20, 30, 50]  # Diferentes valores de top_k
    umbral_values = [1.0, 1.02, 1.05, 1.1]  # Diferentes umbrales

    # Iterar sobre cada combinación de query, top_k y umbral
    for query in queries:
        print(f"\n🔍 **Evaluando query:** {query}")

        for top_k in top_k_values:
            for umbral in umbral_values:
                # Ejecutar la búsqueda en FAISS
                results = store.search(query, top_k=top_k, umbral=umbral)

                # Mostrar resultados
                print(f"✅ FAISS activado con top_k={top_k}, umbral={umbral}. Documentos relevantes: {len(results)}")
                print("\n🔎 Resultados mejorados:")
                print(results)
                print("-" * 80)  # Separador para visualizar mejor las pruebas

    










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

