from fastapi import FastAPI
from api.v1.router import api_router

app = FastAPI(title="Licitaciones API")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "API up and running"}
