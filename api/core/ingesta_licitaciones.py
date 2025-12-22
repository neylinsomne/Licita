import json
import os
import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Importaciones locales
from database.connection import get_db_connection
from pdf_utils import PDFResilientParser
from ai_schemas import TaxonomyPrediction, LicitacionHabilitantes

# CONFIG
client = OpenAI(api_key="TU_API_KEY_AQUI")
print(" Cargando modelo de vectores...")
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

class MasterPipeline:
    def __init__(self):
        self.pdf_parser = PDFResilientParser()

    def run_ingest(self, pdf_path, lic_id_interno):
        print(f"\n PROCESANDO: {lic_id_interno} | Archivo: {pdf_path}")

        # 1. PARSING PDF
        chunks = self.pdf_parser.process(pdf_path)
        print(f" Secciones detectadas: {len(chunks)}")

        # 2. INFERENCIA TAXONOMÍA (Resumen)
        summary_text = " ".join([c['text'] for c in chunks[:3]])[:3000]
        taxonomy = self._infer_taxonomy_llm(summary_text)

        # 3. CONSTRUCCIÓN DEL JSON MAESTRO + VECTORIZACIÓN
        # Estructura base que pediste
        final_json_structure = {
            "licitacion_id": lic_id_interno,
            "metadata_extraccion": {
                "modelo": "gpt-4o", # O el que uses
                "fecha_procesamiento": datetime.datetime.now().isoformat(),
                "taxonomy": taxonomy.model_dump()
            },
            "requisitos_habilitantes": {
                "juridico": [],
                "financiero": [],
                "experiencia": {}
            }
        }

        # Iteramos chunks para llenar la estructura
        for chunk in chunks:
            cat = chunk['category']
            
            # Solo procesamos si es una categoría relevante
            if cat in ["FINANCIERO", "JURIDICO", "EXPERIENCIA"]:
                print(f"   Analizando sección: {chunk['title']} ({cat})")
                extracted = self._extract_requirements_llm(chunk['text'], cat)
                
                if extracted:
                    # A. Procesar JURIDICO / FINANCIERO
                    if extracted.juridico:
                        final_json_structure["requisitos_habilitantes"]["juridico"].extend(
                            self._vectorize_list(extracted.juridico)
                        )
                    if extracted.financiero:
                        final_json_structure["requisitos_habilitantes"]["financiero"].extend(
                            self._vectorize_list(extracted.financiero)
                        )
                    
                    # B. Procesar EXPERIENCIA (Objeto complejo)
                    if extracted.experiencia and not final_json_structure["requisitos_habilitantes"]["experiencia"]:
                         # Convertimos a dict y vectorizamos filtros internos
                         exp_dict = extracted.experiencia.model_dump()
                         # Vectorizar los filtros dentro de experiencia
                         for filtro in exp_dict.get("filtros", []):
                             vec = embedder.encode(filtro["concepto"]).tolist()
                             filtro["vector_concepto"] = vec # INYECCIÓN DEL VECTOR
                             
                         final_json_structure["requisitos_habilitantes"]["experiencia"] = exp_dict

        # 4. VECTOR GLOBAL DEL DOCUMENTO
        doc_vector = embedder.encode(taxonomy.familia_principal + " " + summary_text[:500]).tolist()

        # 5. GUARDAR
        self._save_to_postgres(lic_id_interno, taxonomy, final_json_structure, doc_vector)
        print(f" Licitación {lic_id_interno} guardada y vectorizada.")

    # --- MÉTODOS DE APOYO ---
    def _vectorize_list(self, req_list):
        """
        Recibe lista de objetos Pydantic, convierte a dict y agrega embedding.
        """
        processed = []
        for req in req_list:
            item = req.model_dump()
            #
            vector = embedder.encode(item["concepto"]).tolist()
            item["vector_concepto"] = vector 
            processed.append(item)
        return processed

    def _infer_taxonomy_llm(self, text):
        try:
            return client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "system", "content": "Infiere códigos UNSPSC."}, {"role": "user", "content": text}],
                response_format=TaxonomyPrediction
            ).choices[0].message.parsed
        except:
            return TaxonomyPrediction(codigos_sugeridos=[], familia_principal="N/A", confianza=0)

    def _extract_requirements_llm(self, text, category):
        """Usa el Schema Maestro LicitacionHabilitantes"""
        try:
            return client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": f"Extrae requisitos de tipo {category}. Si es Experiencia, llena el objeto experiencia."},
                    {"role": "user", "content": text}
                ],
                response_format=LicitacionHabilitantes
            ).choices[0].message.parsed
        except Exception as e:
            print(f"Error LLM: {e}")
            return None

    def _save_to_postgres(self, lic_id, taxonomy, full_json, vector):
        conn = get_db_connection()
        cur = conn.cursor()
        sql = """
            INSERT INTO public_licitacion (
                codigo_proceso, entidad, objeto, codigos_unspsc, 
                objeto_vec, metadatos_json
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (codigo_proceso) DO UPDATE 
            SET metadatos_json = EXCLUDED.metadatos_json;
        """
        cur.execute(sql, (
            lic_id, "Privada", taxonomy.familia_principal, 
            taxonomy.codigos_sugeridos, vector, json.dumps(full_json)
        ))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    pipeline = MasterPipeline()
    # pipeline.run_ingest("ejemplo.pdf", "TEST-001")