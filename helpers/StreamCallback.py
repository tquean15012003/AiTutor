import json

from langchain.callbacks.base import AsyncCallbackHandler

from helpers.StreamingResponse import Sender


class StreamManager(AsyncCallbackHandler):
    def __init__(
        self,
        conversation_id: str,
        initial_text: str = "",
        send: Sender = None,
    ):
        super().__init__()
        self.id = conversation_id
        self.text = initial_text
        self.send = send

    async def on_llm_start(self, *args, **kwargs):
        self.text = ""
        data = json.dumps(
            {"answer": "START", "status": "START", "conversation_id": self.id}
        )
        await self.send(f"data: {data}\n\n")

    async def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        data = json.dumps(
            {"answer": self.text, "status": "IN_PROGRESS", "conversation_id": self.id}
        )
        await self.send(f"data: {data}\n\n")

    async def on_llm_end(self, *args, **kwargs):
        data = json.dumps(
            {
                "answer": self.text,
                "status": "COMPLETE_REASONING",
                "conversation_id": self.id,
            }
        )
        await self.send(f"data: {data}\n\n")
