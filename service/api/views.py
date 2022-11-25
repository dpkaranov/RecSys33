# import libraries
from typing import List

from fastapi import APIRouter, Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.auth import is_actual_credentials
from service.api.exceptions import (
    ModelInitializationError,
    ModelNotFoundError,
    UnauthorizedError,
    UserNotFoundError,
)
from service.log import app_logger
from service.models import get_models

# initialize main variables
BEARER = HTTPBearer()
MODELS = get_models()
router = APIRouter()


# custom response model
class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


# func for /health path
@router.get(path="/health", tags=["Health"])
async def root():
    return {"message": "Have a nice day!"}


# func for /reco/{model_name}/{user_id} path
@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(BEARER),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    # checking token
    if not is_actual_credentials(token.credentials):
        raise UnauthorizedError(
            error_message="Token is incorrect"
        )
    # checking user_id
    if user_id > 10**9:
        raise UserNotFoundError(
            error_message=f"User {user_id} not found"
        )
    # checking model_name
    if model_name not in MODELS.keys():
        raise ModelNotFoundError(
            error_message=f"Model {model_name} not found"
        )
    try:
        reco_list = MODELS[model_name].get_reco(user_id)
    except Exception:
        raise ModelInitializationError(
            error_message="Error on model initialization"
        )
    return RecoResponse(user_id=user_id, items=reco_list)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
