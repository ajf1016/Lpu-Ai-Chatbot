import jwt, os
from fastapi import Header, HTTPException

SECRET_KEY = os.getenv("JWT_SECRET")

def get_user_id(authorization: str = Header(...)):
    try:
        token = authorization.split("Bearer ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
