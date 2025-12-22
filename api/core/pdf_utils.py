import fitz  # PyMuPDF
import re
import numpy as np

class PDFResilientParser:
    def process(self, pdf_path):
        """
        Estrategia Maestra:
        1. Intenta extraer por Tabla de Contenido (ToC/Bookmarks) oficial.
        2. Si falla o está vacío, usa la Estrategia Visual (Heurística) que ya tenías.
        3. En ambos casos, detecta y formatea TABLAS automáticamente.
        """
        doc = fitz.open(pdf_path)
        
        # Estrategia 1: Usar Bookmarks del PDF (Si existen)
        toc = doc.get_toc()
        if len(toc) > 2: # Si tiene una estructura decente
            print("📑 Usando estructura oficial del PDF (ToC)...")
            return self._process_by_toc(doc, toc)
        
        # Estrategia 2: Fallback Visual (Tu código original mejorado)
        print("👀 Usando análisis visual heurístico...")
        return self._process_visual(doc)

    # ==========================================
    # LÓGICA DE TABLAS (NUEVO)
    # ==========================================
    def _extract_tables_as_markdown(self, page):
        """
        Busca tablas en la página y las convierte a texto Markdown.
        Retorna: 
        1. El texto Markdown de las tablas.
        2. Una lista de rectángulos (bbox) donde estaban las tablas para NO leer ese texto dos veces.
        """
        tables = page.find_tables() # Función nativa potente de PyMuPDF
        table_text = ""
        table_rects = []
        
        if tables.tables:
            for tab in tables:
                # Convertimos a Markdown porque los LLM (GPT/DeepSeek) lo entienden perfecto
                md = tab.to_markdown() 
                table_text += f"\n\n[TABLA DETECTADA]:\n{md}\n\n"
                table_rects.append(tab.bbox) # Guardamos coordenadas
                
        return table_text, table_rects

    def _is_inside_table(self, block_bbox, table_rects):
        """Verifica si un bloque de texto está dentro de una tabla ya extraída"""
        b_x0, b_y0, b_x1, b_y1 = block_bbox
        for (t_x0, t_y0, t_x1, t_y1) in table_rects:
            # Si el bloque está mayormente contenido en el rect de la tabla
            if b_x0 >= t_x0 and b_y0 >= t_y0 and b_x1 <= t_x1 and b_y1 <= t_y1:
                return True
        return False

    # ==========================================
    # ESTRATEGIA 1: POR BOOKMARKS (ToC)
    # ==========================================
    def _process_by_toc(self, doc, toc):
        chunks = []
        # El ToC viene así: [nivel, titulo, pagina]
        # Iteramos para obtener el rango de páginas entre un título y el siguiente
        for i in range(len(toc)):
            level, title, page_start = toc[i]
            
            # Determinar página final (es el inicio del siguiente capítulo - 1)
            if i + 1 < len(toc):
                page_end = toc[i+1][2] - 1
            else:
                page_end = doc.page_count # Hasta el final
            
            # Corregir índices negativos o ilógicos
            if page_end < page_start - 1: page_end = page_start - 1
            
            # Extraer contenido de ese rango de páginas
            section_text = ""
            for p_num in range(page_start - 1, min(page_end, doc.page_count)):
                page = doc[p_num]
                
                # A. Extraer Tablas primero
                tables_md, table_rects = self._extract_tables_as_markdown(page)
                section_text += tables_md
                
                # B. Extraer Texto (Ignorando lo que ya estaba en tablas)
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if "lines" in b:
                        # Si el bloque está dentro de una tabla, lo saltamos (ya lo tenemos en MD)
                        if self._is_inside_table(b["bbox"], table_rects):
                            continue
                        
                        for l in b["lines"]:
                            for s in l["spans"]:
                                section_text += s["text"] + " "
                    section_text += "\n"

            if section_text.strip():
                chunks.append(self._package(title, section_text))
                
        return chunks

    # ==========================================
    # ESTRATEGIA 2: VISUAL (TU CÓDIGO MEJORADO)
    # ==========================================
    def _process_visual(self, doc):
        chunks = []
        current_title = "RESUMEN_INICIAL"
        buffer_text = ""
        
        for page in doc:
            avg_size = self.get_page_stats(page)
            
            # 1. Extraer Tablas y proteger sus áreas
            tables_md, table_rects = self._extract_tables_as_markdown(page)
            
            # Si hay tablas, las metemos al buffer actual inmediatamente
            if tables_md:
                buffer_text += tables_md
            
            # 2. Leer bloques de texto
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block: continue
                
                # Ignorar texto que ya salió en la tabla Markdown
                if self._is_inside_table(block["bbox"], table_rects):
                    continue

                header_flag = False
                block_text = ""
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text: continue
                        
                        if self.is_header(span, avg_size, text):
                            if buffer_text:
                                chunks.append(self._package(current_title, buffer_text))
                            current_title = text
                            buffer_text = ""
                            header_flag = True
                        else:
                            block_text += text + " "
                
                if not header_flag:
                    buffer_text += block_text + " "

        if buffer_text:
            chunks.append(self._package(current_title, buffer_text))
        return chunks

    # --- UTILIDADES ---
    def get_page_stats(self, page):
        blocks = page.get_text("dict")["blocks"]
        sizes = []
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        sizes.append(s["size"])
        return np.mean(sizes) if sizes else 10.0

    def is_header(self, span, avg_size, text):
        text = text.strip()
        if len(text) < 3 or len(text) > 300: return False
        
        is_large = span["size"] > (avg_size * 1.1)
        is_bold = "bold" in span["font"].lower() or "black" in span["font"].lower()
        is_upper = text.isupper()
        has_numbering = re.match(r"^\d+(\.\d+)*", text)
        keywords = ["ANEXO", "CAPITULO", "FICHA", "PRESUPUESTO", "EXPERIENCIA", "HABILITANTE", "JURIDIC", "FINANCIER"]
        
        if has_numbering and (is_bold or is_upper): return True
        if is_large and is_bold: return True
        if is_upper and any(k in text.upper() for k in keywords): return True
        return False

    def _package(self, title, text):
        cat = "OTRO"
        combined = (title + " " + text[:200]).upper()
        if any(x in combined for x in ["LIQUIDEZ", "PATRIMONIO", "CAPITAL"]): cat = "FINANCIERO"
        elif any(x in combined for x in ["RUP", "JURIDIC", "CONSTITUCION"]): cat = "JURIDICO"
        elif any(x in combined for x in ["TECNIC", "ESPECIFICACION", "ALCANCE"]): cat = "TECNICO"
        return {"title": title, "text": text, "category": cat}