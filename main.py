from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.router import api_router

app = FastAPI(title="Licitaciones API")

# CORS Middleware (Enable for Frontend Dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev convenience (Docker/Localhost)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "API up and running"}
