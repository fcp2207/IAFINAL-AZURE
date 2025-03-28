FROM python:3.11

# Instalar dependencias del sistema necesarias para bitsandbytes y compilación
RUN apt-get update && \
    apt-get install -y git gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 cmake libaio-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Actualiza pip y usa pyproject.toml para instalar dependencias
RUN pip install --upgrade pip && pip install .

# Cache para huggingface
ENV HF_HOME=/app/cache
RUN mkdir -p /app/cache

EXPOSE 8080

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
