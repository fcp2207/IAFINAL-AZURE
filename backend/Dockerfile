# Usa una imagen base de Python 3.11
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del backend al contenedor
COPY . .

# Instala dependencias
RUN pip install --upgrade pip \
    && pip install -e . --no-cache-dir

# Expone el puerto en el que corre FastAPI
EXPOSE 8080

# Comando para iniciar la aplicación FastAPI con Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]

