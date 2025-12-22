import shutil
import os
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from api.orchestrator import TenderPipeline
from database.connection import get_db_connection

router = APIRouter()
# Initialize pipeline once (could also be a dependency)
try:
    pipeline = TenderPipeline()
except Exception as e:
    print(f"Failed to initialize pipeline: {e}")
    pipeline = None

@router.post("/ingest", summary="Ingesta y procesa un PDF de licitación")
async def ingest_licitacion(
    file: UploadFile = File(...), 
    lic_id: str = Form(..., description="ID interno de la licitación")
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline service unavailable")
        
    # Save temp file
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process (Writes to registro_licitaciones, etc.)
        result = pipeline.process_pdf(temp_path, lic_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/", summary="Listar todas las licitaciones registradas")
def list_licitaciones():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, codigo_proceso, entidad, estado_actual, fecha_hora_ingesta 
            FROM registro_licitaciones 
            ORDER BY fecha_hora_ingesta DESC
        """)
        rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "codigo_proceso": r[1],
                "entidad": r[2],
                "estado": r[3],
                "fecha": r[4]
            })
        return result
    finally:
        conn.close()

@router.get("/{lic_id}", summary="Obtener detalle completo de una licitación")
def get_licitacion_details(lic_id: str):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # 1. Get Global Data
        cur.execute("""
            SELECT id, codigo_proceso, metadata_global 
            FROM registro_licitaciones 
            WHERE codigo_proceso = %s
        """, (lic_id,))
        lic_row = cur.fetchone()
        if not lic_row:
            raise HTTPException(status_code=404, detail="Licitación no encontrada")
        
        lic_db_id = lic_row[0]
        
        # 2. Get Sections
        # We join with registro_pdfs to link sections to the process
        cur.execute("""
            SELECT s.titulo_detectado, s.categoria_seccion, s.metadata_extracted
            FROM secciones_documento s
            JOIN registro_pdfs p ON s.pdf_id = p.id
            WHERE p.licitacion_id = %s
        """, (lic_db_id,))
        
        sections = []
        for row in cur.fetchall():
            sections.append({
                "titulo": row[0],
                "categoria": row[1],
                "contenido_extraido": row[2]
            })
            
        return {
            "codigo": lic_row[1],
            "metadata_global": lic_row[2],
            "secciones_procesadas": sections
        }
    finally:
        conn.close()
