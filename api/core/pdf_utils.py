import fitz  # PyMuPDF
import re
import numpy as np

class PDFResilientParser:
    def process(self, pdf_path, use_vision=False):
        """
        Procesa el PDF para extraer texto estructurado.
        NOTA: El parámetro 'use_vision' se ignora aquí intencionalmente.
        La visión (Florence-2) ahora se maneja exclusivamente en el Orchestrator
        para evitar errores de importación circular.
        """
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            
            # Estrategia: Intentar detección visual de estructura (headers/fonts)
            # si el PDF tiene texto seleccionable.
            if doc.page_count > 0:
                chunks = self._process_visual_structure(doc)
            
            # Si la estrategia visual falló o no dio resultados, fallback a página por página
            if not chunks:
                chunks = self._process_simple(doc)
                
            doc.close()
            return chunks
            
        except Exception as e:
            print(f"Error parsing PDF structure: {e}")
            return []

    # --- ESTRATEGIA SIMPLE (Respaldo) ---
    def _process_simple(self, doc):
        chunks = []
        for i, page in enumerate(doc):
            text = self._extract_page_content(page)
            if len(text.strip()) > 50:
                chunks.append(self._package(f"Página {i+1}", text))
        return chunks

    # --- ESTRATEGIA ESTRUCTURADA (Headers por tamaño de fuente) ---
    def _process_visual_structure(self, doc):
        chunks = []
        current_title = "INTRODUCCION"
        buffer_text = ""
        
        for page in doc:
            # Calculamos tamaño promedio de letra para detectar títulos
            avg_size = self._get_page_stats(page)
            page_content, headers = self._extract_page_content_visual(page, avg_size)
            
            # Si hay headers nuevos, cortamos el chunk anterior
            if headers:
                # Guardar lo que llevábamos
                if buffer_text.strip():
                    chunks.append(self._package(current_title, buffer_text))
                
                # Iniciar nuevo bloque con el último header encontrado
                current_title = headers[-1]
                buffer_text = page_content
            else:
                # Si no hay headers, seguimos acumulando en la sección actual
                buffer_text += page_content
                
        # Guardar el último remanente
        if buffer_text.strip():
            chunks.append(self._package(current_title, buffer_text))
            
        return chunks

    # --- UTILIDADES DE EXTRACCIÓN ---
    def _extract_page_content(self, page):
        """Extrae texto plano respetando tablas"""
        tables_md, rects = self._extract_tables_md(page)
        text = tables_md + "\n"
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b and not self._is_inside_table(b["bbox"], rects):
                for l in b["lines"]:
                    for s in l["spans"]:
                        text += s["text"] + " "
        return self._clean_text(text)

    def _extract_page_content_visual(self, page, avg_size):
        """Extrae texto e identifica headers basados en tamaño/negrita"""
        tables_md, rects = self._extract_tables_md(page)
        content = tables_md + "\n"
        detected_headers = []
        
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b and not self._is_inside_table(b["bbox"], rects):
                for l in b["lines"]:
                    for s in l["spans"]:
                        txt = s["text"].strip()
                        if not txt: continue
                        
                        if self._is_header(s, avg_size, txt):
                            detected_headers.append(txt)
                            content += f"\n=== {txt} ===\n"
                        else:
                            content += txt + " "
        return self._clean_text(content), detected_headers

    def _extract_tables_md(self, page):
        """Detecta tablas y las convierte a Markdown"""
        tables = page.find_tables()
        text_md = ""
        rects = []
        if tables:
            for tab in tables:
                text_md += f"\n[TABLA DETECTADA]:\n{tab.to_markdown()}\n"
                rects.append(tab.bbox)
        return text_md, rects

    def _is_inside_table(self, bbox, table_rects):
        for r in table_rects:
            # Chequeo simple de intersección/contención
            if bbox[0] >= r[0] and bbox[1] >= r[1] and bbox[2] <= r[2] and bbox[3] <= r[3]:
                return True
        return False

    def _get_page_stats(self, page):
        try:
            sizes = [s["size"] for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
            return np.mean(sizes) if sizes else 10.0
        except:
            return 10.0

    def _is_header(self, span, avg, text):
        if len(text) < 3 or len(text) > 200: return False
        is_large = span["size"] > avg * 1.15
        is_bold = "bold" in span["font"].lower()
        
        keywords = ["ANEXO", "CAPITULO", "SECCION", "OBJETO", "PRESUPUESTO", "EXPERIENCIA", "HABILITANTE", "ESPECIFICACION"]
        is_keyword = any(k in text.upper() for k in keywords)
        
        return (is_large and is_bold) or (is_bold and is_keyword) or (is_large and text.isupper())

    def _clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def _package(self, title, text):
        cat = "GENERAL"
        t_upper = (title + text[:200]).upper()
        
        if any(x in t_upper for x in ["LIQUIDEZ", "FINANCIER", "CAPITAL", "ENDEUDAMIENTO"]): cat = "FINANCIERO"
        elif any(x in t_upper for x in ["RUP", "JURIDIC", "CONSTITUCION", "REPRESENTA"]): cat = "JURIDICO"
        elif any(x in t_upper for x in ["TECNIC", "ESPECIFICACION", "ALCANCE", "MEMORIA"]): cat = "TECNICO"
        elif "EXPERIENCIA" in t_upper or "CONTRATOS" in t_upper: cat = "EXPERIENCIA"
            
        return {"title": title, "text": text, "category": cat}