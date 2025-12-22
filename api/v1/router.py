from fastapi import APIRouter
from api.v1.endpoints import licitaciones, pipelines
# Commenting out others if they are broken, or we should fix imports if they exist.
# Assuming they exist based on list_dir, so we fix the import path.
from api.v1.endpoints import auth, empresas, opportunities, storage

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(empresas.router, prefix="/empresas", tags=["empresas"])
api_router.include_router(licitaciones.router, prefix="/licitaciones", tags=["licitaciones"])
api_router.include_router(opportunities.router, prefix="/opportunities", tags=["opportunities"])
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(storage.router, prefix="/storage", tags=["storage"])
