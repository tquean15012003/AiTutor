import logging

from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form

from helpers.QueryAnswerer import QueryAnswerer

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/")
async def answer_query(
    conversation_id: str = Form(...),
    query: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    logger.info("Answer query api is called.")
    query_answerer = QueryAnswerer()
    answer = await query_answerer.answer_query(
        conversation_id=conversation_id, query=query, image=image
    )
    return answer
