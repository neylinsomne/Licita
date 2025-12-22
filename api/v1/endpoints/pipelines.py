from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_pipelines():
    return {"message": "Pipeline logic now integrated into Orquestador (TenderPipeline)."}
