from fastapi import FastAPI
from app.auth import router as auth_router
from app.chat import router as chat_router

app = FastAPI()

# Register routes
app.include_router(auth_router)
app.include_router(chat_router)
