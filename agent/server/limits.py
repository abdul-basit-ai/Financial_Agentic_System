from __future__ import annotations

import os

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        cap = (
            int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))) + 512 * 1024
            if scope.get("path") == "/api/v1/documents"
            else 1024 * 1024
        )
        received = 0

        async def bounded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    raise HTTPException(413, "The request exceeds the size limit.")
            return message

        await self.app(scope, bounded_receive, send)
