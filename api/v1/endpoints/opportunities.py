from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_opportunities():
    return {"message": "Not implemented yet. Please refactor."}