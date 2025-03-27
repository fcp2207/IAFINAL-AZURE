# Usa una imagen de Python 3.11
FROM python:3.11

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos del backend al contenedor
COPY . .

# Instala UV y dependencias desde pyproject.toml
RUN pip install uv && uv pip install -e .

# Exponer el puerto del backend
EXPOSE 8080

# Comando para correr el backend con Uvicorn
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
