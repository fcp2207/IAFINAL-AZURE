# Usa una imagen base con Python
FROM python:3.11

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del frontend al contenedor
COPY . .

# Instala UV y las dependencias del proyecto
RUN pip install uv && uv pip install 

# Expone el puerto en el que se ejecuta Chainlit
EXPOSE 8000

# Comando para iniciar Chainlit
CMD ["uv", "run", "chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
