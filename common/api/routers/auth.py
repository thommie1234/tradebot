"""Auth router — login with API key, get JWT."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.auth import create_token
from api.config import API_KEY, JWT_EXPIRY_DAYS
from api.models.schemas import LoginRequest, LoginResponse

router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if req.api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    token = create_token()
    return LoginResponse(token=token, expires_in_days=JWT_EXPIRY_DAYS)
