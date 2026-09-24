"""Bounded installed-query access through TigerGraph's official MCP server."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _decode_result(content: str) -> list[dict[str, Any]]:
    if not content.startswith("```json\n"):
        raise ValueError("TigerGraph MCP returned an unexpected response")
    payload = json.loads(content.split("```json\n", 1)[1].split("\n```", 1)[0])
    if not payload.get("success"):
        raise RuntimeError(f"TigerGraph MCP query failed: {payload.get('error', 'unknown error')}")
    rows = payload["data"]["result"]
    if not isinstance(rows, list):
        raise ValueError("TigerGraph MCP installed query returned no row list")
    return rows


async def _run(query_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = os.environ.get("TG_MCP_URL", "http://127.0.0.1:8001/mcp/")
    async with streamable_http_client(url) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool(
                "tigergraph__run_installed_query",
                arguments={"query_name": query_name, "params": params},
            )
            if result.is_error or not result.content:
                raise RuntimeError(f"TigerGraph MCP call failed: {query_name}")
            return _decode_result(result.content[0].text)


def run_installed_query(query_name: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"results": asyncio.run(_run(query_name, params))}
