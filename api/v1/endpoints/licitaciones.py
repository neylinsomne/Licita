import shutil
import os
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from api.orchestrator import TenderPipeline

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
        
        # Process
        result = pipeline.process_pdf(temp_path, lic_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/", summary="Listar licitaciones (Placeholder)")
def list_licitaciones():
    return {"message": "Use database connection to list tenders directly or implement repository layer."}
