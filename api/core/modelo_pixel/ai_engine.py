import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURACIÓN GLOBAL ---
MODEL_ID = "microsoft/Florence-2-large"

# CORRECCIÓN 1: Usar la variable de entorno o fallback a local
# Esto asegura que lea exactamente donde 'descargar_modelo.py' guardó los archivos
MODEL_CACHE_DIR = os.getenv("HF_HOME", "./model_cache")

# Variables Globales
_MODEL = None
_PROCESSOR = None
_DEVICE = None

def _init_model():
    """Inicializa el modelo solo si no está cargado."""
    global _MODEL, _PROCESSOR, _DEVICE
    
    if _MODEL is not None:
        return # Ya está cargado

    print("-" * 30)
    if torch.cuda.is_available():
        _DEVICE = "cuda"
        print(f"✅ MOTOR IA: GPU Detectada: {torch.cuda.get_device_name(0)}")
        dtype = torch.float16 
    else:
        _DEVICE = "cpu"
        dtype = torch.float32
        print("⚠️ MOTOR IA: Usando CPU (Lento).")
    print("-" * 30)

    print(f"⏳ Cargando modelo desde: {MODEL_CACHE_DIR}...")
    try:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True, 
            torch_dtype=dtype,
            cache_dir=MODEL_CACHE_DIR
        ).to(_DEVICE)
        
        _PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True, 
            cache_dir=MODEL_CACHE_DIR
        )
        print("🚀 Modelo cargado y listo en memoria.")
    except Exception as e:
        print(f"❌ Error fatal cargando el modelo: {e}")
        raise e

def run_ocr_inference(image_path, task="<OCR>"):
    # 1. Asegurar que el modelo esté vivo
    if _MODEL is None:
        _init_model()

    if not os.path.exists(image_path):
        return {"error": "Imagen no encontrada"}

    try:
        image = Image.open(image_path).convert("RGB")
        
        # Procesamiento
        inputs = _PROCESSOR(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        if _DEVICE == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        generated_ids = _MODEL.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )

        generated_text = _PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        parsed_answer = _PROCESSOR.post_process_generation(
            generated_text, 
            task=task, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer

    except Exception as e:
        return {"error": str(e)}

# --- CORRECCIÓN 2: Bloque de Ejecución Principal ---
# Esto evita que el Docker se cierre instantáneamente y prueba que todo funcione.
if __name__ == "__main__":
    print("🏁 Iniciando Test de Integridad del Contenedor...")
    
    # 1. Forzamos la carga del modelo
    _init_model()

    # 2. Generamos una imagen dummy para probar (así no necesitas montar volúmenes para probar)
    print("🖼️ Generando imagen de prueba interna...")
    img = Image.new('RGB', (200, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10,10), "HOLA MUNDO", fill=(0,0,0)) # Escribe texto negro
    img.save("test_interno.jpg")

    # 3. Corremos inferencia
    print("🧠 Ejecutando inferencia de prueba...")
    resultado = run_ocr_inference("test_interno.jpg")
    
    print("\nRESULTADO DE PRUEBA:")
    print(resultado)
    print("\n✅ El sistema está operativo. El contenedor finalizará ahora.")