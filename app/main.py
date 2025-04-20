from fastapi import FastAPI, Depends, Header, HTTPException
from auth import router as auth_router
from pydantic import BaseModel
from rag_engine import get_qa_chain
from supabase_client import supabase
import jwt, os
from typing import Optional

app = FastAPI()
app.include_router(auth_router)

SECRET_KEY = os.getenv("JWT_SECRET")

def get_user_id(authorization: str = Header(...)):
    try:
        token = authorization.split("Bearer ")[1]  # Extract token from "Bearer <token>"
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

@app.post("/chat/start")
def start_new_chat(title: Optional[str] = "Untitled Chat", user_id=Depends(get_user_id)):
    chat = supabase.table("chats").insert({"user_id": user_id, "title": title}).execute().data[0]
    return {"chat_id": chat["id"], "title": chat["title"]}


class MessageRequest(BaseModel):
    query: str
    
@app.post("/chat/{chat_id}")
def send_message(chat_id: str, request: MessageRequest, user_id=Depends(get_user_id)):
    # Store user's message
    # supabase.table("messages").insert({
    #     "chat_id": chat_id,
    #     "role": "user",
    #     "content": request.query
    # }).execute()

    # Run through RAG
    qa = get_qa_chain()
    result = qa.invoke({"query": request.query})["result"]

    # Store assistant's reply
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "assistant",
        "query": request.query,
        "answer": result
    }).execute()

    return {"response": result}

@app.get("/chat/{chat_id}/history")
def get_chat_history(chat_id: str, user_id=Depends(get_user_id)):
    messages = supabase.table("messages").select("*").eq("chat_id", chat_id).order("timestamp").execute().data
    return {"messages": messages}


@app.get("/chats")
def get_all_chats(user_id=Depends(get_user_id)):
    chats = supabase.table("chats").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
    return {"chats": chats}