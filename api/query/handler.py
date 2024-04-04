import json
import logging

from fastapi import APIRouter
from typing import Awaitable, Callable

from api.query.model import AnswerQueryRequest
from helpers.QueryAnswerer import QueryAnswerer
from helpers.StreamCallback import StreamManager
from helpers.StreamingResponse import ChatOpenAIStreamingResponse, Sender

router = APIRouter()

logger = logging.getLogger(__name__)


def send_message(
    conversation_id: str, query: str, image=None
) -> Callable[[Sender], Awaitable[None]]:
    async def generate(send: Sender):
        callback = StreamManager(conversation_id=conversation_id, send=send)
        query_answerer = QueryAnswerer(stream_callback=[callback])
        answer = await query_answerer.answer_query(
            conversation_id=conversation_id, query=query, image=image
        )
        data = json.dumps(
            {
                "answer": answer,
                "status": "COMPLETE",
                "conversation_id": conversation_id,
            }
        )
        await send(f"data: {data}\n\n")

    return generate


@router.post("/")
async def answer_query(body: AnswerQueryRequest):
    conversation_id = body.conversation_id
    query = body.query

    return ChatOpenAIStreamingResponse(
        send_message(conversation_id=conversation_id, query=query),
        media_type="text/event-stream",
    )
