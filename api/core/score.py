from pydantic import BaseModel, Field
from typing import List

# --- MODELOS DE DATOS ---
class TaxonomyPrediction(BaseModel):
    codigos_sugeridos: List[str] = Field(..., description="Códigos UNSPSC (8 dígitos)")
    familia_principal: str = Field(..., description="Familia del servicio")
    confianza: float

class RequisitoItem(BaseModel):
    concepto: str = Field(..., description="Nombre estandarizado (ej: Indice Liquidez)")
    operador: str = Field(..., description=">=, <=, =")
    valor_numerico: float
    unidad: str
    cita_textual: str

class ExtraccionHabilitante(BaseModel):
    lista_requisitos: List[RequisitoItem]

# --- LÓGICA DE INFERENCIA ---
def infer_taxonomy(text, client):
    """Predice códigos UNSPSC para licitaciones privadas"""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06", # Cambiar modelo según tu proveedor
            messages=[
                {"role": "system", "content": "Eres un clasificador UNSPSC. Infiere los códigos."},
                {"role": "user", "content": text[:3000]} # Limitamos contexto
            ],
            response_format=TaxonomyPrediction
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"⚠️ Error Taxonomía: {e}")
        return TaxonomyPrediction(codigos_sugeridos=[], familia_principal="Desconocido", confianza=0.0)

def extract_requirements(text, category, client):
    """Extrae JSON estructurado de requisitos"""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": f"Extrae requisitos {category} habilitantes."},
                {"role": "user", "content": text}
            ],
            response_format=ExtraccionHabilitante
        )
        return completion.choices[0].message.parsed
    except Exception:
        return None