from pydantic import BaseModel


class AnswerQueryRequest(BaseModel):
    conversation_id: str
    query: str
