import torch
from transformers import AutoModelForCausalLM, AutoProcessor

def descargar_y_cachear():
    print("⏳ Iniciando descarga de Florence-2 (esto puede tardar)...")
    
    model_id = "microsoft/Florence-2-large"
    
    # 1. Descargar y guardar en cache
    # IMPORTANTE: attn_implementation="eager" arregla el error de SDPA en PyTorch < 2.4
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager" 
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    
    print("✅ Modelo descargado y verificado exitosamente.")

if __name__ == "__main__":
    descargar_y_cachear()