# Usa una imagen base con Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos al contenedor
COPY . .

# Instala dependencias desde pyproject.toml
RUN pip install --upgrade pip \
    && pip install .

# Expone el puerto de Chainlit
EXPOSE 8000

# Comando para iniciar Chainlit
CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]

