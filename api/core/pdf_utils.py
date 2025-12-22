import fitz  # PyMuPDF
import re
import numpy as np
import os
from PIL import Image
import io

# Import Florence Engine (Lazy loaded inside methods if needed to avoid startup overhead if not used)
# Assuming relative import works from api/core
try:
    from api.core.modelo_pixel.ai_engine import run_ocr_inference
except ImportError:
    # Fallback for docker structure if path is slightly different or if running from root
    try:
        from core.modelo_pixel.ai_engine import run_ocr_inference
    except ImportError:
        print("Warning: Could not import ai_engine. Visual features disabled.")
        def run_ocr_inference(*args, **kwargs): return {}

class PDFResilientParser:
    def process(self, pdf_path, use_vision=True):
        """
        Procesa el PDF.
        Args:
            pdf_path: Ruta al archivo.
            use_vision: Si True, usa Florence-2 para enriquecer la metadata visualmente.
        Returns:
            List[Dict]: Chunks procesados con texto y (opcionalmente) descripción visual.
        """
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        
        # 1. VISUAL ENRICHMENT (Florence-2)
        # Generamos una "capa de metadatos visuales" por página antes de fragmentar
        visual_metadata = {}
        if use_vision:
            print("👁️ Ejecutando análisis visual (Florence-2)...")
            visual_metadata = self._analyze_visual_content(doc)

        chunks = []
        # 2. ESTRATEGIA DE ESTRUCTURA (ToC vs Visual Heuristic)
        if len(toc) > 2:
            print("📑 Usando estructura oficial del PDF (ToC)...")
            chunks = self._process_by_toc(doc, toc)
        else:
            print("👀 Usando análisis visual heurístico (Fallback)...")
            chunks = self._process_visual(doc)
            
        # 3. MERGE: Inyectar descripción visual al chunk correspondiente
        # Como los chunks pueden abarcar varias páginas, agregamos la info de las páginas que tocan.
        # En este MVP, simplemente adjuntamos el metadata global al primer chunk o lo retornamos aparte.
        # El Orchestrator decidirá dónde guardarlo (probablemente en registro_pdfs -> metadata_archivo)
        
        return chunks, visual_metadata

    # --- FLORENCE INTEGRATION ---
    def _analyze_visual_content(self, doc):
        """
        Renderiza páginas a imágenes y corre Florence-2.
        Retorna un dict: { page_num: { 'caption': str, 'ocr_full': str } }
        """
        metadata = {}
        # Limitamos a primeras 5 páginas para eficiencia en demo
        pages_to_scan = list(range(min(5, doc.page_count))) 
        
        for p_idx in pages_to_scan:
            page = doc[p_idx]
            pix = page.get_pixmap(dpi=150) # Render a 150 DPI
            img_data = pix.tobytes("png")
            
            # Guardamos temporalmente para que lo lea ai_engine (que espera path)
            temp_img = f"temp_page_{p_idx}.png"
            with open(temp_img, "wb") as f:
                f.write(img_data)
                
            try:
                # 1. Caption Detallado (Describe tablas, gráficos, logos)
                captionor = run_ocr_inference(temp_img, task="<MORE_DETAILED_CAPTION>")
                
                # Check for errors
                desc = ""
                if isinstance(captionor, dict) and "error" not in captionor:
                    desc = captionor.get("<MORE_DETAILED_CAPTION>", "")
                
                mask_text = f"Page {p_idx+1}: {desc}"
                metadata[p_idx] = {"visual_description": desc}
                print(f"   📸 {mask_text[:60]}...")
            except Exception as e:
                print(f"Error visual page {p_idx}: {e}")
            finally:
                if os.path.exists(temp_img):
                    os.remove(temp_img)
                    
        return metadata

    # --- LÓGICA DE TABLAS ---
    def _extract_tables_md(self, page):
        """Extrae tablas y devuelve texto Markdown + Coordenadas"""
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
            if bbox[0] >= r[0] and bbox[1] >= r[1] and bbox[2] <= r[2] and bbox[3] <= r[3]:
                return True
        return False

    # --- PROCESADORES TEXTO ---
    def _process_by_toc(self, doc, toc):
        chunks = []
        for i in range(len(toc)):
            title = toc[i][1]
            start_page = toc[i][2] - 1
            
            # Validación de rangos
            if i + 1 < len(toc):
                end_page = toc[i+1][2] - 1
            else:
                end_page = doc.page_count

            start_page = max(0, min(start_page, doc.page_count - 1))
            end_page = max(start_page, min(end_page, doc.page_count))

            section_text = ""
            for p_idx in range(start_page, end_page):
                section_text += self._extract_page_content(doc[p_idx])
            
            if section_text.strip():
                chunks.append(self._package(title, section_text))
        return chunks

    def _process_visual(self, doc):
        chunks = []
        current_title = "RESUMEN_INICIAL"
        buffer_text = ""
        
        for page in doc:
            avg_size = self._get_page_stats(page)
            page_content, headers = self._extract_page_content_visual(page, avg_size)
            
            buffer_text += page_content
            
            if headers: 
                chunks.append(self._package(current_title, buffer_text))
                current_title = headers[-1] # Último título visto
                buffer_text = ""
                
        if buffer_text:
            chunks.append(self._package(current_title, buffer_text))
        return chunks

    def _extract_page_content(self, page):
        tables_md, rects = self._extract_tables_md(page)
        text = tables_md + "\n"
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b and not self._is_inside_table(b["bbox"], rects):
                for l in b["lines"]:
                    for s in l["spans"]:
                        text += s["text"] + " "
        return text

    def _extract_page_content_visual(self, page, avg_size):
        tables_md, rects = self._extract_tables_md(page)
        content = tables_md + "\n"
        detected_headers = []
        
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b and not self._is_inside_table(b["bbox"], rects):
                for l in b["lines"]:
                    for s in l["spans"]:
                        txt = s["text"].strip()
                        if self._is_header(s, avg_size, txt):
                            detected_headers.append(txt)
                            content += f"\n=== {txt} ===\n"
                        else:
                            content += txt + " "
        return content, detected_headers

    def _get_page_stats(self, page):
        sizes = [s["size"] for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
        return np.mean(sizes) if sizes else 10.0

    def _is_header(self, span, avg, text):
        is_large = span["size"] > avg * 1.1
        is_bold = "bold" in span["font"].lower()
        has_num = re.match(r"^\d+(\.\d+)*", text)
        keywords = ["ANEXO", "CAPITULO", "FICHA", "PRESUPUESTO", "EXPERIENCIA", "HABILITANTE", "ESPECIFICACION"]
        return (has_num and (is_bold or text.isupper())) or (is_large and is_bold) or (text.isupper() and any(k in text for k in keywords))

    def _package(self, title, text):
        cat = "OTRO"
        t_upper = (title + text[:100]).upper()
        if any(x in t_upper for x in ["LIQUIDEZ", "FINANCIER", "CAPITAL"]): cat = "FINANCIERO"
        elif any(x in t_upper for x in ["RUP", "JURIDIC", "CONSTITUCION"]): cat = "JURIDICO"
        elif any(x in t_upper for x in ["TECNIC", "ESPECIFICACION", "ALCANCE", "MEMORIA"]): cat = "TECNICO"
        if "EXPERIENCIA" in t_upper: cat = "EXPERIENCIA"
            
        return {"title": title, "text": text, "category": cat}