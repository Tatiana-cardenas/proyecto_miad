FROM python:3.10-slim

WORKDIR /app

# Copiarrequirements.txt
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando por defecto para correr Streamlit
CMD ["streamlit", "run", "tablero.py", "--server.port=8501", "--server.address=0.0.0.0"]
