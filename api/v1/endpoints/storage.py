from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_storage():
    return {"message": "Not implemented yet. Please refactor."}
