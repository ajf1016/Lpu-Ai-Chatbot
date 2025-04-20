from fastapi import FastAPI
from auth import router as auth_router
from chat import router as chat_router

app = FastAPI()

# Register routes
app.include_router(auth_router)
app.include_router(chat_router)
