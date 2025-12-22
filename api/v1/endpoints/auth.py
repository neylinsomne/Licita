from fastapi import APIRouter

router = APIRouter()

@router.post("/login")
def login():
    return {"message": "Not implemented yet. Please refactor."}

@router.post("/register")
def register():
    return {"message": "Not implemented yet. Please refactor."}
