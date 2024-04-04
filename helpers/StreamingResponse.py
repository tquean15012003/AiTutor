from starlette.types import Send
from fastapi.responses import StreamingResponse
from typing import Awaitable, Callable, Optional, Union

Sender = Callable[[Union[str, bytes]], Awaitable[None]]


class ChatOpenAIStreamingResponse(StreamingResponse):
    """Streaming response for openai chat model, inheritance from StreamingResponse."""

    def __init__(
        self,
        generate: Callable[[Sender], Awaitable[None]],
        status_code: int = 200,
        media_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            content=iter(()), status_code=status_code, media_type=media_type
        )
        self.generate = generate

    async def stream_response(self, send: Send) -> None:
        """Rewrite stream_response to send response to client."""
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        async def send_chunk(chunk: Union[str, bytes]):
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        # send body to client
        await self.generate(send_chunk)

        # send empty body to client to close connection
        await send({"type": "http.response.body", "body": b"", "more_body": False})
