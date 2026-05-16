#!/bin/bash

# Script para desplegar en una instancia EC2 de AWS

# 1. Actualizar el sistema
sudo dnf update -y || sudo yum update -y

# 2. Instalar Docker
sudo dnf install -y docker || sudo yum install -y docker

# 3. Iniciar el servicio de Docker y habilitarlo para que inicie en el arranque
sudo systemctl start docker
sudo systemctl enable docker

# 4. Agregar el usuario actual al grupo docker para no tener que usar sudo en cada comando
sudo usermod -aG docker ec2-user

# 5. Construir la imagen de Docker
# docker build -t tablero-cafe .

# 6. Correr el contenedor exponiendo el puerto 8501 al puerto 80
# docker run -d -p 80:8501 --name tablero-app --restart always tablero-cafe
