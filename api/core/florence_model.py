import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os

# --- VERIFICACIÓN DE GPU ---
print("-" * 30)
if torch.cuda.is_available():
    print(f" ÉXITO: CUDA detectado.")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    DEVICE = "cuda"
else:
    print(" AVISO: No se detectó GPU. Se usará CPU (lento).")
    DEVICE = "cpu"
print("-" * 30)

MODEL_ID = "microsoft/Florence-2-large"


model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def procesar_documento(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        return "Error: No encuentro la imagen."
    
    image = Image.open(ruta_imagen).convert("RGB")
    prompt = "<OCR>" 
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

print("Analizando imagen...")
resultado = procesar_documento("documento.jpg")
print("\nRESULTADO:\n")
print(resultado)