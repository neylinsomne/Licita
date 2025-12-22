from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. ESQUEMAS DE DATOS (SCHEMAS)
# ==========================================

class TaxonomyPrediction(BaseModel):
    codigos_sugeridos: List[str] = Field(..., description="Códigos UNSPSC (8 dígitos) inferidos")
    familia_principal: str = Field(..., description="Nombre de la familia principal del servicio")
    confianza: float


class RequisitoItem(BaseModel):
    id_req: Optional[str] = Field(None, description="ID autogenerado si aplica")
    concepto: str = Field(..., description="Nombre normalizado (ej: Indice de Liquidez)")
    operador: str = Field(..., description="Operador matemático: >=, <=, =, CONTAINS")
    # Union permite que sea numero (1.5), booleano (true) o string ("Cumple")
    valor_requerido: Union[float, bool, str] = Field(..., description="El valor objetivo a cumplir")
    unidad: str = Field(..., description="Unidad: veces, %, SMMLV, boolean, años")
    fuente_texto: str = Field(..., description="Fragmento del texto original para auditoría")
    vector_concepto: Optional[List[float]] = None 

class FiltroExperiencia(RequisitoItem):
    valores_lista: Optional[List[str]] = Field(None, description="Lista de códigos o valores aceptados")
    nivel_coincidencia: Optional[str] = Field(None, description="ej: familia, clase, segmento")

class ExperienciaContainer(BaseModel):
    regla_general: str = Field(..., description="Resumen narrativo de la regla de experiencia")
    filtros: List[FiltroExperiencia] = Field(default_factory=list)

class LicitacionHabilitantes(BaseModel):
    juridico: List[RequisitoItem] = Field(default_factory=list)
    financiero: List[RequisitoItem] = Field(default_factory=list)
    experiencia: Optional[ExperienciaContainer] = None


# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN (IA WRAPPERS)
# ==========================================

def infer_taxonomy(text, client):
    """Predice códigos UNSPSC para licitaciones privadas"""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06", 
            messages=[
                {"role": "system", "content": "Eres un clasificador UNSPSC experto. Infiere los códigos y familia."},
                {"role": "user", "content": text[:3000]} 
            ],
            response_format=TaxonomyPrediction
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f" Error Taxonomía: {e}")
        return TaxonomyPrediction(codigos_sugeridos=[], familia_principal="Desconocido", confianza=0.0)

def extract_requirements(text, category, client):
    """Extrae la estructura compleja de requisitos"""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": f"Extrae requisitos {category} habilitantes. Si es Experiencia, estructura los filtros detalladamente."},
                {"role": "user", "content": text}
            ],
            response_format=LicitacionHabilitantes
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error Extracción {category}: {e}")
        return None

# ==========================================
# 3. MOTOR DE CÁLCULO DE SCORE (MATCHING)
# ==========================================

def calcular_match_total(licitacion_db, empresa_perfil):
    """
    Calcula el score final (0-100) combinando 3 factores.
    
    Args:
        licitacion_db: Dict con datos de la BD (incluyendo vectores y jsonb).
        empresa_perfil: Dict con datos de la empresa (vectores, indicadores).
        
    Returns:
        (float, list): Score final y lista de alertas/razones.
    """
    
    # 1. FILTROS HABILITANTES (Hard Constraints)
    # Si falla uno crítico, el score se penaliza severamente o es 0.
    cumple_financiero, alertas_fin = _check_financiero(
        licitacion_db.get('metadatos_json', {}).get('requisitos_habilitantes', {}).get('financiero', []),
        empresa_perfil.get('indicadores', {})
    )
    
    if not cumple_financiero:
        return 0.0, alertas_fin + ["Descalificado por Financiero"]

    # 2. SCORE SEMÁNTICO (Vectores) - Peso 40%
    # Comparamos el vector "Objeto Licitación" vs "Perfil Empresa"
    vec_lic = np.array(licitacion_db['objeto_vec']).reshape(1, -1)
    vec_emp = np.array(empresa_perfil['perfil_vec']).reshape(1, -1)
    score_sem = cosine_similarity(vec_lic, vec_emp)[0][0] # 0 a 1
    
    # 3. SCORE TAXONÓMICO (Códigos UNSPSC) - Peso 40%
    
    codes_lic = set(licitacion_db.get('codigos_unspsc', []))
    codes_emp = set(empresa_perfil.get('codigos_unspsc', []))
    
    if codes_lic:
        coincidencias = len(codes_lic.intersection(codes_emp))
        score_tax = coincidencias / len(codes_lic)
    else:
        score_tax = 0.5
    # 4. BONUS FINANCIERO - Peso 20%

    score_fin = 1.0 # Ya validamos que cumple arriba
    
    # CÁLCULO FINAL PONDERADO
    W_SEM = 0.4
    W_TAX = 0.4
    W_FIN = 0.2
    
    final_score = (score_sem * W_SEM) + (score_tax * W_TAX) + (score_fin * W_FIN)
    
    return round(final_score * 100, 2), alertas_fin + ["Habilitado"]

# --- Helpers de Verificación ---

def _check_financiero(requisitos_lic, indicadores_emp):
    """Valida reglas financieras dinámicas > < ="""
    alertas = []
    cumple_todo = True
    
    for req in requisitos_lic:
        concepto = req['concepto'] # Ej: "Indice de Liquidez"
        operador = req['operador']
        valor_req = float(req['valor_requerido'])
        
        # Buscamos el valor en la empresa (normalizando claves si es necesario)
        # Asumimos que indicadores_emp tiene claves similares
        val_emp = indicadores_emp.get(concepto, 0)
        
        cumple_local = True
        if operador == '>=':
            cumple_local = val_emp >= valor_req
        elif operador == '<=':
            cumple_local = val_emp <= valor_req
        elif operador == '=':
            cumple_local = val_emp == valor_req
            
        if not cumple_local:
            cumple_todo = False
            alertas.append(f"Falla {concepto}: Tienes {val_emp}, piden {operador} {valor_req}")
            
    return cumple_todo, alertas