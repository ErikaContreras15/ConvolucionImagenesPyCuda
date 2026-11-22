
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

# Python 3.12 de Ubuntu 24.04 + herramientas para compilar pycuda
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Entorno virtual propio (Ubuntu 24.04 no permite pip en el Python del sistema)
RUN python3 -m venv /opt/venv

WORKDIR /app
# Primero solo las dependencias: si el código cambia, esta capa se reutiliza
COPY Practica03/requirements.txt Practica03/requirements.txt
RUN pip install -r Practica03/requirements.txt

# Código: kernel y utilidades de la P02 (sin cambios) + servicio de la P03
COPY Practica02/ Practica02/
COPY Practica03/ Practica03/

WORKDIR /app/Practica03
EXPOSE 5000

# Servidor de producción: sin depurador ni recarga automática.
# 1 worker = 1 proceso = 1 contexto CUDA; 8 hilos HTTP que esperan en la cola del hilo GPU.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "300", "--bind", "0.0.0.0:5000", "app:app"]
