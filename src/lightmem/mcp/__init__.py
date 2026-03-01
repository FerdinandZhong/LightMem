"""
LightMem MCP Server

Usage:
    # With environment variables (recommended):
    export OPENAI_API_KEY="your-api-key"
    lightmem-mcp

    # Via uvx:
    uvx --from "git+https://github.com/zjunlp/LightMem.git[mcp]" lightmem-mcp
"""

from lightmem.mcp.server import main, mcp

__all__ = ["main", "mcp"]
