from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import supabase
from passlib.hash import bcrypt
import jwt
import os

router = APIRouter()
SECRET_KEY = os.getenv("JWT_SECRET")

# Define a Pydantic model for the signup request body
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

@router.post("/signup")
def signup(request: SignupRequest):
    hashed_pw = bcrypt.hash(request.password)
    existing = supabase.table("users").select("*").eq("email", request.email).execute().data
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    user = supabase.table("users").insert({
        "name": request.name,
        "email": request.email,
        "password": hashed_pw
    }).execute().data[0]
    token = jwt.encode({"user_id": user["id"]}, SECRET_KEY, algorithm="HS256")
    return {"token": token}

# Define a Pydantic model for the login request body
class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(request: LoginRequest):
    user = supabase.table("users").select("*").eq("email", request.email).execute().data
    if not user or not bcrypt.verify(request.password, user[0]["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"user_id": user[0]["id"]}, SECRET_KEY, algorithm="HS256")
    return {"token": token}