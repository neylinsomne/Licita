from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_empresas():
    return {"message": "Not implemented yet. Please refactor."}
