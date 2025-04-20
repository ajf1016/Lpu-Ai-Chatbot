from fastapi import APIRouter, Depends
from typing import Optional
from supabase_client import supabase
from dependencies import get_user_id
from rag_engine import get_qa_chain
from schemas import MessageRequest

router = APIRouter()

@router.post("/chat/start")
def start_new_chat(title: Optional[str] = "Untitled Chat", user_id=Depends(get_user_id)):
    chat = supabase.table("chats").insert({"user_id": user_id, "title": title}).execute().data[0]
    return {"chat_id": chat["id"], "title": chat["title"]}

@router.post("/chat/{chat_id}")
def send_message(chat_id: str, request: MessageRequest, user_id=Depends(get_user_id)):
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

@router.get("/chat/{chat_id}/history")
def get_chat_history(chat_id: str, user_id=Depends(get_user_id)):
    messages = supabase.table("messages").select("*").eq("chat_id", chat_id).order("timestamp").execute().data
    return {"messages": messages}

@router.get("/chats")
def get_all_chats(user_id=Depends(get_user_id)):
    chats = supabase.table("chats").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
    return {"chats": chats}
