from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.hash import bcrypt
from jose import jwt
import os
from app.db.database import SessionLocal
from app.db.models import User

router = APIRouter()
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret")  # fallback if not in env

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/signup")
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    hashed_pw = bcrypt.hash(request.password)
    new_user = User(name=request.name, email=request.email, password=hashed_pw)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = jwt.encode({"user_id": str(new_user.id)}, SECRET_KEY, algorithm="HS256")
    return {"token": token}


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not bcrypt.verify(request.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode({"user_id": str(user.id)}, SECRET_KEY, algorithm="HS256")
    return {"token": token}
