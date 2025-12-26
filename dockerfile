# 1. Base Image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
# Compatibilidad para GPUs modernas (incluida serie 50xx vía PTX)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" 
#ENV HF_HOME=/app/model_cache
ENV HF_HOME=/hf_cache

WORKDIR /app

# 2. Dependencias del Sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 3. Dependencias de Python (requirements.txt)
COPY requirements.txt .
# Esto instalará dependencias genéricas y un PyTorch "viejo" que luego borraremos
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 4. LIMPIEZA
# Borramos el torch estable incompatible
RUN pip uninstall -y torch torchvision torchaudio

# 5. INSTALACIÓN DE TORCH NIGHTLY (Soporte Blackwell/RTX 50xx)
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# 6. INSTALACIÓN DE COMPLEMENTOS (Vision + Timm)
# - timm: requerido por Florence-2
# - torchvision: instalado con --no-deps para evitar conflicto de versión con torch nightly
RUN pip install timm && \
    pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps

# 7. HORNEADO DEL MODELO
# Copiamos el script y lo ejecutamos una sola vez
COPY api/core/modelo_pixel/descargar_modelo.py .
RUN python descargar_modelo.py

# 8. Copia del resto del código (Descomenta esto cuando estés listo para copiar tu app completa)
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]