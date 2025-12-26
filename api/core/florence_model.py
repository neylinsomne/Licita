import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# Configuración
MODEL_ID = "microsoft/Florence-2-large"
# Usamos variable de entorno o caché local por defecto
MODEL_CACHE_DIR = os.getenv("HF_HOME", "/app/model_cache")

_MODEL = None
_PROCESSOR = None
_DEVICE = None

def _init_model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL: return

    print("--- INICIALIZANDO MOTOR IA (RTX 5060) ---")
    if torch.cuda.is_available():
        _DEVICE = "cuda"
        print(f" GPU Detectada: {torch.cuda.get_device_name(0)}")
        dtype = torch.float16
    else:
        _DEVICE = "cpu"
        print("⚠️ Usando CPU (Lento)")
        dtype = torch.float32

    try:
        # ATENCIÓN: attn_implementation="eager" es lo que evita el error 500 en tu tarjeta
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True, 
            torch_dtype=dtype,
            cache_dir=MODEL_CACHE_DIR,
            attn_implementation="eager" 
        ).to(_DEVICE)
        
        _PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True, 
            cache_dir=MODEL_CACHE_DIR
        )
        print(" Florence-2 Listo.")
    except Exception as e:
        print(f" Error Fatal IA: {e}")
        raise e

def run_ocr_inference(image_path, task="<OCR>"):
    if not _MODEL: _init_model()
    if not os.path.exists(image_path): return {"error": "Imagen no encontrada"}
    
    try:
        image = Image.open(image_path).convert("RGB")
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
        
        text = _PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return _PROCESSOR.post_process_generation(text, task=task, image_size=image.size)
    except Exception as e:
        return {"error": str(e)}