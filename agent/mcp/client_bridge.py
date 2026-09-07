"""MCP Client Bridge adapter.

Connects to the FastMCP server instance in-process or over stdio/SSE transports,
enabling LangGraph nodes to discover and execute MCP tools dynamically.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from pydantic import BaseModel, Field

from agent.mcp.server import mcp


class DiscoveredTool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPClientBridge:
    """In-process and transport client bridge interfacing with the MCP Server."""

    def __init__(self, server_instance: Any | None = None) -> None:
        self._server = server_instance or mcp

    async def list_tools_async(self) -> list[DiscoveredTool]:
        """Asynchronously discovers all tools exposed by the MCP server."""
        tools_list = await self._server.list_tools()
        return [
            DiscoveredTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.parameters if hasattr(t, "parameters") else {},
            )
            for t in tools_list
        ]

    def list_tools(self) -> list[DiscoveredTool]:
        """Synchronously discovers tools."""
        return asyncio.run(self.list_tools_async())

    async def call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatches tool invocation through the MCP server interface.

        Handles both MCP 1.x return shapes: the low-level tuple
        ``(content_blocks, structured)`` and any list-of-blocks wrapper.
        """
        result = await self._server.call_tool(name=name, arguments=arguments)

        # MCP >=1.x low-level server returns a (content_blocks, structured_dict) tuple
        if isinstance(result, tuple) and len(result) >= 1:
            content_blocks = result[0]
        elif isinstance(result, list):
            content_blocks = result
        else:
            content_blocks = [result]

        if content_blocks:
            first_block = content_blocks[0]
            raw_text = getattr(first_block, "text", None)
            if raw_text is None:
                raw_text = str(first_block)
        else:
            raw_text = ""

        try:
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else {"raw_output": parsed}
        except Exception:
            return {"raw_output": raw_text}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Synchronous dispatch wrapper for call_tool_async."""
        return asyncio.run(self.call_tool_async(name, arguments))

    async def read_resource_async(self, uri: str) -> str:
        """Reads a passive context stream from the MCP server.

        MCP read_resource returns a list of ReadResourceContents blocks;
        concatenates text content from all blocks.
        """
        content = await self._server.read_resource(uri)
        if isinstance(content, list):
            parts = []
            for block in content:
                text = getattr(block, "content", None) or getattr(block, "text", None)
                if text is None:
                    text = str(block)
                parts.append(str(text))
            return "\n".join(parts)
        return str(content)

    def read_resource(self, uri: str) -> str:
        """Synchronous wrapper for read_resource_async."""
        return asyncio.run(self.read_resource_async(uri))