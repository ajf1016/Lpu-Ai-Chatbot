from pydantic import BaseModel

class MessageRequest(BaseModel):
    query: str
