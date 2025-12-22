import os
import datetime
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from database.connection import get_db_connection
from api.core.pdf_utils import PDFResilientParser
from api.core.ai_schema import TaxonomyPrediction, LicitacionHabilitantes

class TenderPipeline:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key)
        # Using a lightweight model for embedding by default
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.pdf_parser = PDFResilientParser()

    def process_pdf(self, pdf_path: str, lic_id_interno: str):
        """
        Orchestrates the processing of a PDF file:
        1. Parsing
        2. Taxonomy Inference
        3. Requirement Extraction
        4. Vectorization
        5. Saving to DB
        """
        print(f"\n🚀 STARTING PIPELINE: {lic_id_interno} | File: {pdf_path}")

        # 1. PARSING PDF
        chunks = self.pdf_parser.process(pdf_path)
        print(f"📄 Generated {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No text extracted from PDF")

        # 2. INFERENCIA TAXONOMÍA (Resumen)
        # Take first 3 chunks as summary context
        summary_text = " ".join([c['text'] for c in chunks[:3]])[:3000]
        taxonomy = self._infer_taxonomy_llm(summary_text)

        # 3. CONSTRUCCIÓN DEL JSON MAESTRO + VECTORIZACIÓN
        final_json_structure = {
            "licitacion_id": lic_id_interno,
            "metadata_extraccion": {
                "modelo": "gpt-4o", 
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
            
            if cat in ["FINANCIERO", "JURIDICO", "EXPERIENCIA"]:
                print(f"   🔍 Analyzing section: {chunk['title']} ({cat})")
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
                    # Merge strategy: if we find experience, we take the first valid one or merge? 
                    # For now simplistically taking the first non-empty one or merging fields could be complex.
                    # We'll stick to logic similar to original: populate if empty.
                    if extracted.experiencia:
                         current_exp = final_json_structure["requisitos_habilitantes"]["experiencia"]
                         if not current_exp:
                             # Convertimos a dict y vectorizamos filtros internos
                             exp_dict = extracted.experiencia.model_dump()
                             # Vectorizar los filtros dentro de experiencia
                             for filtro in exp_dict.get("filtros", []):
                                 vec = self.embedder.encode(filtro["concepto"]).tolist()
                                 filtro["vector_concepto"] = vec 
                             final_json_structure["requisitos_habilitantes"]["experiencia"] = exp_dict

        # 4. VECTOR GLOBAL DEL DOCUMENTO
        # Combined vector: Taxonomy + Summary
        doc_vector = self.embedder.encode(taxonomy.familia_principal + " " + summary_text[:500]).tolist()

        # 5. GUARDAR
        self._save_to_postgres(lic_id_interno, taxonomy, final_json_structure, doc_vector)
        print(f"✅ Tender {lic_id_interno} processed and saved.")
        return final_json_structure

    # --- HELPER METHODS ---
    def _vectorize_list(self, req_list):
        processed = []
        for req in req_list:
            item = req.model_dump()
            vector = self.embedder.encode(item["concepto"]).tolist()
            item["vector_concepto"] = vector 
            processed.append(item)
        return processed

    def _infer_taxonomy_llm(self, text):
        try:
            return self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "system", "content": "Infiere códigos UNSPSC."}, {"role": "user", "content": text}],
                response_format=TaxonomyPrediction
            ).choices[0].message.parsed
        except Exception as e:
            print(f"Taxonomy Inference Error: {e}")
            return TaxonomyPrediction(codigos_sugeridos=[], familia_principal="N/A", confianza=0)

    def _extract_requirements_llm(self, text, category):
        try:
            return self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": f"Extrae requisitos de tipo {category}. Si es Experiencia, llena el objeto experiencia."},
                    {"role": "user", "content": text}
                ],
                response_format=LicitacionHabilitantes
            ).choices[0].message.parsed
        except Exception as e:
            print(f"Extraction Error LLM: {e}")
            return None

    def _save_to_postgres(self, lic_id, taxonomy, full_json, vector):
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            # Ensure table exists (basic check, ideally handled by migration script)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public_licitacion (
                    codigo_proceso TEXT PRIMARY KEY,
                    entidad TEXT,
                    objeto TEXT,
                    codigos_unspsc TEXT[],
                    objeto_vec vector(384),
                    metadatos_json JSONB
                );
            """)

            sql = """
                INSERT INTO public_licitacion (
                    codigo_proceso, entidad, objeto, codigos_unspsc, 
                    objeto_vec, metadatos_json
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (codigo_proceso) DO UPDATE 
                SET metadatos_json = EXCLUDED.metadatos_json,
                    objeto_vec = EXCLUDED.objeto_vec,
                    codigos_unspsc = EXCLUDED.codigos_unspsc,
                    objeto = EXCLUDED.objeto;
            """
            cur.execute(sql, (
                lic_id, "Privada", taxonomy.familia_principal, 
                taxonomy.codigos_sugeridos, vector, json.dumps(full_json)
            ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"DB Error: {e}")
            raise e
        finally:
            conn.close()
