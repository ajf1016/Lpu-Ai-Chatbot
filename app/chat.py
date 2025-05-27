from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional
from uuid import uuid4
from app.dependencies import get_user_id
from app.schemas import MessageRequest
from app.rag_engine import get_qa_chain
from app.db.database import SessionLocal
from app.db.models import Chat, Message

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/chat/start")
def start_new_chat(
    title: Optional[str] = "Untitled Chat",
    user_id=Depends(get_user_id),
    db: Session = Depends(get_db)
):
    chat = Chat(id=uuid4(), user_id=user_id, title=title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": str(chat.id), "title": chat.title}


@router.post("/chat/{chat_id}")
def send_message(
    chat_id: str,
    request: MessageRequest,
    user_id=Depends(get_user_id),
    db: Session = Depends(get_db)
):
    # Run query through RAG
    qa = get_qa_chain()
    result = qa.invoke({"query": request.query})["result"]

    # Save assistant's message
    message = Message(
        id=uuid4(),
        chat_id=chat_id,
        role="assistant",
        query=request.query,
        answer=result
    )
    db.add(message)
    db.commit()

    return {"response": result}


@router.get("/chat/{chat_id}/history")
def get_chat_history(
    chat_id: str,
    user_id=Depends(get_user_id),
    db: Session = Depends(get_db)
):
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()
    return {"messages": [ 
        {
            "id": str(m.id),
            "role": m.role,
            "query": m.query,
            "answer": m.answer,
            "timestamp": m.timestamp.isoformat()
        } for m in messages
    ]}


@router.get("/chats")
def get_all_chats(
    user_id=Depends(get_user_id),
    db: Session = Depends(get_db)
):
    chats = db.query(Chat).filter(Chat.user_id == user_id).order_by(desc(Chat.created_at)).all()
    return {"chats": [
        {
            "id": str(c.id),
            "title": c.title,
            "created_at": c.created_at.isoformat()
        } for c in chats
    ]}
