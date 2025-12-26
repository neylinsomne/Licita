import os
import json
import io
import fitz  # PyMuPDF
from PIL import Image

# --- IMPORTACIONES ---
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from api.core.pdf_utils import PDFResilientParser
from database.connection import get_db_connection
from api.core.modelo_pixel.ai_engine import analizar_imagen_con_florence

class TenderPipeline:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(" WARNING: GOOGLE_API_KEY not found.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                print(" Using Google GenAI Client (Default/Beta).")
            except Exception as e:
                print(f" Error init Gemini: {e}")
                self.client = None
        
        self.model_name = "gemini-2.5-flash" 

        print(" Loading embedding model (all-mpnet-base-v2)...")
        try:
            self.embedder = SentenceTransformer('all-mpnet-base-v2', device='cpu')
            self.embedder.encode("warmup")
            print(" Embeddings cargados y listos.")
        except Exception as e:
            print(f" Error crítico en Embeddings: {e}")
            raise e
        
        self.parser = PDFResilientParser()

    def process_pdf(self, pdf_path: str, lic_id_interno: str):
        print(f"\nSTARTING PIPELINE: {lic_id_interno} | File: {pdf_path}")

        # ---------------------------------------------------------
        # PASO 0: VISIÓN COMPUTACIONAL (Florence-2)
        # ---------------------------------------------------------
        print(" Ejecutando análisis visual (Florence-2)...")
        visual_metadata = {}
        contexto_visual_global = ""
        
        try:
            doc = fitz.open(pdf_path)
            # Procesamos todas las páginas
            for page_num, page in enumerate(doc):
                try:
                    pix = page.get_pixmap(dpi=150)
                    img_data = pix.tobytes("png")
                    
                    # CORRECCIÓN DE VISIÓN: Convertir a RGB siempre
                    # Esto arregla el error: 'NoneType' object has no attribute 'shape'
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    
                    # Llamada a la GPU
                    descripcion = analizar_imagen_con_florence(image)
                    
                    # Guardar
                    page_key = f"page_{page_num + 1}"
                    visual_metadata[page_key] = descripcion
                    contexto_visual_global += f"[Página {page_num+1} Análisis Visual]: {descripcion}\n"
                    
                except Exception as e_vision:
                    print(f"  Error visión en página {page_num}: {e_vision}")
            
            doc.close()
            print(f"  Análisis visual completado en {len(visual_metadata)} páginas.")
            
        except Exception as e:
            print(f"Error crítico abriendo PDF para visión: {e}")

        # ---------------------------------------------------------
        # PASO 1: PARSING DE TEXTO
        # ---------------------------------------------------------
        chunks = self.parser.process(pdf_path, use_vision=False)
        if isinstance(chunks, tuple): chunks = chunks[0]

        if not chunks:
            raise ValueError(f"No text extracted from {pdf_path}")
        print(f" Extracted {len(chunks)} text chunks.")
        
        # ---------------------------------------------------------
        # PASO 2: TAXONOMÍA GLOBAL (Gemini + Visión)
        # ---------------------------------------------------------
        resumen_texto = " ".join([c.get('text', '') for c in chunks[:10]])[:3000]
        contexto_total = f"RESUMEN VISUAL:\n{contexto_visual_global[:1500]}\n\nTEXTO INICIAL:\n{resumen_texto}"
        
        taxonomy = self._infer_taxonomy_gemini(contexto_total)
        print(f" Taxonomy: {taxonomy.get('familia_principal', 'Unknown')}")
        
        # ---------------------------------------------------------
        # PASO 3: GUARDADO EN BASE DE DATOS
        # ---------------------------------------------------------
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            
            # A. INSERT LICITACION
            cur.execute("""
                INSERT INTO registro_licitaciones (codigo_proceso, entidad, estado_actual, metadata_global)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (codigo_proceso) DO UPDATE 
                SET estado_actual = 'PROCESANDO',
                    metadata_global = EXCLUDED.metadata_global
                RETURNING id;
            """, (
                lic_id_interno, 
                "Entidad Pendiente", 
                "PROCESANDO", 
                json.dumps(taxonomy)
            ))
            lic_db_id = cur.fetchone()[0]

            # B. INSERT PDF
            file_meta = {
                "size_bytes": os.path.getsize(pdf_path), 
                "page_count_est": len(chunks),
                "visual_content": visual_metadata 
            }
            
            cur.execute("""
                INSERT INTO registro_pdfs (licitacion_id, nombre_archivo, ruta_almacenamiento, metadata_archivo)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (lic_db_id, os.path.basename(pdf_path), pdf_path, json.dumps(file_meta)))
            pdf_db_id = cur.fetchone()[0]

            # C. SECCIONES & VECTORES
            for chunk in chunks:
                cat = chunk.get('category', 'GENERAL')
                page_num = chunk.get('page', 1)
                title = chunk.get('title', f"Página {page_num}")
                text = chunk.get('text', '')

                # Insertar Sección
                cur.execute("""
                    INSERT INTO secciones_documento (pdf_id, titulo_detectado, categoria_seccion, metadata_extracted)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                """, (pdf_db_id, title, cat, '{}'))
                sec_id = cur.fetchone()[0]

                # Vectorizar Texto
                text_vec = self.embedder.encode(text[:800]).tolist()
                cur.execute("""
                    INSERT INTO nodos_vectorizados (seccion_id, tipo_nodo, contenido_texto, embedding_vec)
                    VALUES (%s, 'CHUNK_TEXTO', %s, %s)
                """, (sec_id, text, text_vec))

                # Extraer Requisitos
                if cat in ["FINANCIERO", "JURIDICO", "EXPERIENCIA", "TECNICO"]:
                    info_visual_pagina = visual_metadata.get(f"page_{page_num}", "")
                    extracted = self._extract_requirements_gemini(text, cat, info_visual_pagina)
                    
                    if extracted:
                        cur.execute("UPDATE secciones_documento SET metadata_extracted = %s WHERE id = %s", 
                                    (json.dumps(extracted), sec_id))
                        
                        for key in ['juridico', 'financiero']:
                            for item in extracted.get(key, []):
                                self._insert_node(cur, sec_id, f'REQUISITO_{key.upper()}', item)
                        
                        exp = extracted.get('experiencia', {})
                        if exp and 'filtros' in exp:
                            for item in exp['filtros']:
                                self._insert_node(cur, sec_id, 'REQUISITO_EXPERIENCIA', item)

            cur.execute("UPDATE registro_licitaciones SET estado_actual = 'INDEXADO' WHERE id = %s", (lic_db_id,))
            conn.commit()
            return {"status": "success", "licitacion_id": lic_db_id}

        except Exception as e:
            conn.rollback()
            print(f"Error DB Transaction: {e}")
            raise e
        finally:
            conn.close()

    def _insert_node(self, cur, sec_id, node_type, item_dict):
        concept = item_dict.get('concepto', 'N/A')
        if not concept: concept = "Indefinido"
        vec = self.embedder.encode(str(concept)[:500]).tolist()
        cur.execute("""
            INSERT INTO nodos_vectorizados (seccion_id, tipo_nodo, contenido_texto, metadata_nodo, embedding_vec)
            VALUES (%s, %s, %s, %s, %s)
        """, (sec_id, node_type, concept, json.dumps(item_dict), vec))



    # ---------------------------------------------------------
    # MÉTODOS GEMINI (MODO COMPATIBILIDAD V1)
    # ---------------------------------------------------------
    def _infer_taxonomy_gemini(self, text):
        if not self.client: return {"familia_principal": "No API Key"}
        
        # Pedimos JSON explícitamente en el prompt
        prompt = f"""
        Actúa como experto en licitaciones. Analiza:
        {text[:4000]}
        
        Responde SOLO JSON válido (sin markdown):
        {{ "familia_principal": "Nombre", "codigos_sugeridos": ["123"], "confianza": 0.9 }}
        """
        try:
            # CORRECCIÓN: Sin 'config'. Esto evita el error 400.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            # Limpieza manual
            txt = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception as e:
            print(f"Gemini Tax Error: {e}")
            return {"familia_principal": "Error IA"}

    def _extract_requirements_gemini(self, text, category, visual_context=""):
        if not self.client: return {}

        prompt = f"""
        Extrae requisitos {category}. 
        Contexto Visual: {visual_context}
        Texto: {text}
        
        Responde SOLO JSON válido (sin markdown):
        {{
            "juridico": [], "financiero": [], "experiencia": {{ "filtros": [] }}
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            txt = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception as e:
            print(f"Gemini Ext Error: {e}")
            return {}

if __name__ == "__main__":
    if os.path.exists("test.pdf"):
        pipeline = TenderPipeline()
        pipeline.process_pdf("test.pdf", "TEST001")