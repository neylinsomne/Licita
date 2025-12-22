from pydantic import BaseModel, Field
from typing import List, Optional, Union

# --- MODELO PARA INFERENCIA DE CÓDIGOS UNSPSC ---
class TaxonomyPrediction(BaseModel):
    codigos_sugeridos: List[str] = Field(..., description="Códigos UNSPSC inferidos")
    familia_principal: str = Field(..., description="Nombre de la familia principal")
    confianza: float

# --- MODELOS PARA REQUISITOS HABILITANTES ---

# 1. Item Simple (Jurídico / Financiero)
class RequisitoItem(BaseModel):
    id_req: Optional[str] = Field(None, description="ID autogenerado ej: FIN-01")
    concepto: str = Field(..., description="Nombre normalizado ej: Indice de Liquidez")
    operador: str = Field(..., description=">=, <=, =, CONTAINS")
    valor_requerido: Union[float, bool, str] = Field(..., description="El valor objetivo")
    unidad: str = Field(..., description="Unidad: veces, %, SMMLV, boolean")
    fuente_texto: str = Field(..., description="Fragmento original del PDF")
    condicional_extra: Optional[str] = None
    valor_normalizado_cop: Optional[float] = None

# 2. Estructura de Experiencia (Más compleja)
class FiltroExperiencia(RequisitoItem):
    valores_lista: Optional[List[str]] = Field(None, description="Para listas de códigos UNSPSC")
    nivel_coincidencia: Optional[str] = Field(None, description="ej: tercer_nivel")

class ExperienciaContainer(BaseModel):
    regla_general: str = Field(..., description="Resumen de la regla de experiencia")
    filtros: List[FiltroExperiencia]

# 3. Estructura Maestra (Salida del LLM)
class LicitacionHabilitantes(BaseModel):
    juridico: List[RequisitoItem] = Field(default_factory=list)
    financiero: List[RequisitoItem] = Field(default_factory=list)
    experiencia: Optional[ExperienciaContainer] = None

