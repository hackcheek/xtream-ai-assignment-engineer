from fastapi import APIRouter
from challenge3.src.api.routes import ch1, ch2


api_router = APIRouter()
api_router.include_router(ch1.router, tags=['ch1', 'predict'])

# NotImplemented
# api_router.include_router(ch2.router, prefix='/ch2', tags=['ch2 (Not Implemented)', 'predict'])
