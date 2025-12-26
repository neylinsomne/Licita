import sys
import os
import torch

print("----- 1. VERIFICACIÓN DE ENTORNO -----")
print(f"Python: {sys.version}")
try:
    import google.genai
    print(f"Librería google-genai (NUEVA): INSTALADA ({google.genai.__version__})")
except ImportError:
    print("Librería google-genai (NUEVA): NO ENCONTRADA")

try:
    import google.generativeai
    print(f"Librería google.generativeai (VIEJA): INSTALADA ({google.generativeai.__version__})")
except ImportError:
    print("Librería google.generativeai (VIEJA): NO ENCONTRADA")

print("\n----- 2. PRUEBA DE IMPORTACIÓN AI_ENGINE (FLORENCE) -----")
try:
    # Intentamos importar tal cual lo hace tu app
    from api.core.modelo_pixel.ai_engine import analizar_imagen_con_florence, model, processor
    print("✅ Importación EXITOSA.")
    print(f"Model config: {model.config._name_or_path}")
    print(f"Processor: {type(processor)}")
    print(f"Dtype del modelo: {model.dtype}")
except Exception as e:
    print("❌ ERROR FATAL IMPORTANDO AI_ENGINE:")
    import traceback
    traceback.print_exc()

print("\n----- 3. PRUEBA DE CONEXIÓN GEMINI (CLIENTE V1) -----")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ NO HAY API KEY.")
else:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("Intentando generar contenido con 'gemini-1.5-flash'...")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Di 'Hola Gemini' si funcionas."
        )
        print(f"✅ RESPUESTA GEMINI: {response.text}")
    except Exception as e:
        print(f"❌ ERROR GEMINI:\n{e}")