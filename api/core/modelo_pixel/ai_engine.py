from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import os

# --- CONFIGURACIÓN ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model_id = "microsoft/Florence-2-large"

print(f"--- FLORENCE ENGINE INIT ---")
print(f"Device: {device}")
print(f"Dtype objetivo: {dtype}")

try:
    # Cargamos con 'eager' para evitar errores de SDPA
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✅ Modelo cargado.")
except Exception as e:
    print(f"❌ ERROR CARGA MODELO: {e}")
    model = None

def analizar_imagen_con_florence(image_path_or_obj, task_prompt="<MORE_DETAILED_CAPTION>", text_input=None):
    if model is None: return {"error": "Model not loaded"}

    print("--- INICIO DEBUG FLORENCE ---")
    try:
        # 1. IMAGEN
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj).convert("RGB")
        else:
            image = image_path_or_obj.convert("RGB")
        
        print(f"1. Imagen OK: {image.size} Mode={image.mode}")

        # 2. PROMPT
        prompt = task_prompt + (text_input if text_input else "")
        print(f"2. Prompt: '{prompt}'")

        # 3. PROCESSOR
        # Importante: images=[image]
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        print(f"3. Keys del Processor: {inputs.keys()}")
        
        if "pixel_values" not in inputs:
            print("❌ ERROR: pixel_values no existe en inputs")
            return ""
            
        if inputs["pixel_values"] is None:
            print("❌ ERROR: pixel_values es None")
            return ""

        # 4. TENSORES Y TIPOS (Aquí está la clave)
        pv = inputs["pixel_values"]
        ids = inputs["input_ids"]
        
        print(f"   -> Pixel Values Shape Original: {pv.shape}, Dtype: {pv.dtype}")
        print(f"   -> Input IDs Shape Original: {ids.shape}, Dtype: {ids.dtype}")

        # MOVER A GPU Y CASTEAR
        # pixel_values debe ser float16 (si usas cuda)
        # input_ids debe ser long (int64), NUNCA float
        
        inputs["pixel_values"] = pv.to(device, dtype)
        inputs["input_ids"] = ids.to(device) # IDs se quedan como enteros (Long)

        print(f"   -> PV en GPU: {inputs['pixel_values'].device}, Type: {inputs['pixel_values'].dtype}")
        print(f"   -> IDs en GPU: {inputs['input_ids'].device}, Type: {inputs['input_ids'].dtype}")

        # 5. GENERATE (El punto de quiebre)
        print("5. Ejecutando model.generate()...")
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,         
            # early_stopping=False # <--- BORRAR: No sirve con num_beams=1
        )
        
        print(f"6. Generación OK. IDs shape: {generated_ids.shape}")

        # 6. DECODE
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"7. Texto Raw: {generated_text[:50]}...")

        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        if isinstance(parsed_answer, dict) and task_prompt in parsed_answer:
            return parsed_answer[task_prompt]
            
        return parsed_answer

    except Exception as e:
        print(f"❌ EXCEPCIÓN EN FLORENCE: {e}")
        import traceback
        traceback.print_exc() # Esto nos dirá la línea exacta dentro de la librería
        return ""

def run_ocr_inference(image, task="<MORE_DETAILED_CAPTION>"):
    return analizar_imagen_con_florence(image, task_prompt=task)