import os
import json
import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Adjust imports based on your folder structure
from database.connection import get_db_connection
from api.core.pdf_utils import PDFResilientParser
# Ensure this matches your file name (ai_schemas.py vs ai_schema.py)
from api.core.ai_schemas import TaxonomyPrediction, LicitacionHabilitantes

class TenderPipeline:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(" WARNING: OPENAI_API_KEY not set. LLM calls will fail.")
        
        self.client = OpenAI(api_key=api_key)
        
        # CRITICAL FIX: Matching DB dimensions (768)
        print("⏳ Loading embedding model (all-mpnet-base-v2)...")
        self.embedder = SentenceTransformer('all-mpnet-base-v2') 
        self.pdf_parser = PDFResilientParser()

    def process_pdf(self, pdf_path: str, lic_id_interno: str):
        """
        Full Pipeline: PDF -> Layout Analysis -> LLM Extraction -> Graph Nodes -> Postgres
        """
        print(f"\nSTARTING PIPELINE: {lic_id_interno} | File: {pdf_path}")

        # 1. PARSING & CHUNKING (VISUAL ENRICHMENT)
        # Now expecting a tuple: (chunks, visual_metadata)
        try:
            chunks, visual_metadata = self.pdf_parser.process(pdf_path, use_vision=True)
        except Exception as e:
            # Fallback for old signature or error
            print(f"Vision Pipeline Warning: {e}. Falling back to standard parse.")
            try:
                # If process() returned just chunks in a previous version or failure mode
                res = self.pdf_parser.process(pdf_path, use_vision=False)
                if isinstance(res, tuple):
                    chunks, visual_metadata = res
                else:
                    chunks = res
                    visual_metadata = {}
            except Exception as e2:
                 raise ValueError(f"CRITICAL: Failed to parse PDF: {e2}")

        if not chunks:
            raise ValueError(f" No text extracted from {pdf_path}")
        print(f"📄 Extracted {len(chunks)} semantic sections.")
        
        # 2. GLOBAL METADATA INFERENCE (Taxonomy)
        summary_text = " ".join([c['text'] for c in chunks[:3]])[:3000]
        taxonomy = self._infer_taxonomy_llm(summary_text)
        print(f"🏷️ Taxonomy Detected: {taxonomy.familia_principal}")
        
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # --- A. LEVEL 1: TENDER REGISTRY (Upsert) ---
            cur.execute("""
                INSERT INTO registro_licitaciones (codigo_proceso, entidad_estatal, estado_actual, metadata_global)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (codigo_proceso) DO UPDATE 
                SET estado_actual = 'PROCESANDO',
                    metadata_global = EXCLUDED.metadata_global
                RETURNING id;
            """, (
                lic_id_interno, 
                "Empresa Privada", # Placeholder if unknown
                "PROCESANDO", 
                json.dumps(taxonomy.model_dump())
            ))
            lic_db_id = cur.fetchone()[0]

            # --- B. LEVEL 2: PDF REGISTRY ---
            # Save visual metadata here
            file_meta = {
                "size_bytes": os.path.getsize(pdf_path), 
                "page_count_est": len(chunks),
                "visual_content": visual_metadata # Florence-2 results
            }
            
            cur.execute("""
                INSERT INTO registro_pdfs (licitacion_id, nombre_archivo, ruta_almacenamiento, metadata_archivo)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (
                lic_db_id, 
                os.path.basename(pdf_path), 
                pdf_path, 
                json.dumps(file_meta)
            ))
            pdf_db_id = cur.fetchone()[0]

            # --- C. LEVEL 3 & 4: SECTIONS & NODES ---
            for chunk in chunks:
                cat = chunk['category']
                title = chunk['title']
                text = chunk['text']

                # 3.1 Insert Section (Context Container)
                cur.execute("""
                    INSERT INTO secciones_documento (pdf_id, titulo_detectado, categoria_seccion, metadata_extracted)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (
                    pdf_db_id, title, cat, json.dumps({}) # Updated later
                ))
                sec_id = cur.fetchone()[0]

                # 3.2 Create "Raw Text" Node (For Semantic Search)
                text_vec = self.embedder.encode(text[:800]).tolist()
                cur.execute("""
                    INSERT INTO nodos_vectorizados (seccion_id, tipo_nodo, contenido_texto, embedding_vec)
                    VALUES (%s, 'CHUNK_TEXTO', %s, %s)
                """, (sec_id, text, text_vec))

                # 3.3 Intelligent Extraction (For relevant sections)
                if cat in ["FINANCIERO", "JURIDICO", "EXPERIENCIA", "TECNICO"]:
                    extracted = self._extract_requirements_llm(text, cat)
                    
                    if extracted:
                        # Update Section Metadata (JSONB)
                        cur.execute("""
                            UPDATE secciones_documento 
                            SET metadata_extracted = %s 
                            WHERE id = %s
                        """, (json.dumps(extracted.model_dump()), sec_id))

                        # 3.4 Create "Atom" Nodes (For Graph Logic)
                        for item in extracted.juridico:
                            self._insert_node(cur, sec_id, 'REQUISITO_JURIDICO', item)
                        
                        for item in extracted.financiero:
                            self._insert_node(cur, sec_id, 'REQUISITO_FINANCIERO', item)
                            
                        if extracted.experiencia and extracted.experiencia.filtros:
                            for item in extracted.experiencia.filtros:
                                self._insert_node(cur, sec_id, 'REQUISITO_EXPERIENCIA', item)

            # --- D. FINALIZE ---
            cur.execute("UPDATE registro_licitaciones SET estado_actual = 'INDEXADO' WHERE id = %s", (lic_db_id,))
            conn.commit()
            print(f"Tender {lic_id_interno} fully indexed.")
            return {"status": "success", "licitacion_id": lic_db_id}

        except Exception as e:
            conn.rollback()
            print(f"DB Pipeline Error: {e}")
            raise e
        finally:
            conn.close()

    # --- HELPER: Node Insertion ---
    def _insert_node(self, cur, sec_id, node_type, item_obj):
        """Helper to vectorize a concept and insert it as a Graph Node"""
        # Vectorize the concept name (e.g., "Liquidity Index")
        vec = self.embedder.encode(item_obj.concepto).tolist()
        
        cur.execute("""
            INSERT INTO nodos_vectorizados 
            (seccion_id, tipo_nodo, contenido_texto, metadata_nodo, embedding_vec)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            sec_id, 
            node_type, 
            item_obj.concepto, # The 'Name' of the node
            json.dumps(item_obj.model_dump()), # The 'Properties' of the node
            vec # The 'Position' in latent space
        ))

    # --- LLM WRAPPERS ---
    def _infer_taxonomy_llm(self, text):
        try:
            return self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "system", "content": "Infiere códigos UNSPSC."}, {"role": "user", "content": text}],
                response_format=TaxonomyPrediction
            ).choices[0].message.parsed
        except Exception as e:
            print(f"Taxonomy Error: {e}")
            return TaxonomyPrediction(codigos_sugeridos=[], familia_principal="Desconocido", confianza=0)

    def _extract_requirements_llm(self, text, category):
        try:
            return self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": f"Extrae requisitos {category} habilitantes."},
                    {"role": "user", "content": text}
                ],
                response_format=LicitacionHabilitantes
            ).choices[0].message.parsed
        except Exception:
            return None

if __name__ == "__main__":
    if os.path.exists("test.pdf"):
        pipeline = TenderPipeline()
        pipeline.process_pdf("test.pdf", "TEST-PIPELINE-001")