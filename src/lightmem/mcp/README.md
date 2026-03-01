# LightMem MCP Server

Launch LightMem as an MCP server via `uvx` for seamless integration with MCP-compatible clients.

## Quick Start

### MCP Client Configuration

Add to your MCP client config (e.g., Claude Desktop, Cursor):

```json
{
  "mcpServers": {
    "lightmem": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light", "lightmem-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LIGHTMEM_LLM_MODEL` | No | `gpt-4o-mini` | LLM model for memory operations |
| `LIGHTMEM_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `LIGHTMEM_EMBEDDING_DIMS` | No | `1536` | Embedding dimensions |
| `LIGHTMEM_DATA_PATH` | No | `./lightmem_data` | Path for Qdrant storage |
| `LIGHTMEM_COLLECTION_NAME` | No | `lightmem_memory` | Qdrant collection name |
| `LIGHTMEM_CONFIG_PATH` | No | - | Path to custom config JSON file |

### Config File

Alternatively, provide a JSON config file via `LIGHTMEM_CONFIG_PATH` or `--config` flag. See [example.json](example.json) for the API-only configuration template.

## Available Tools

| Tool | Description |
|------|-------------|
| `get_timestamp` | Get current timestamp |
| `add_memory` | Add user/assistant message pair to memory |
| `retrieve_memory` | Query memories by natural language |
| `offline_update` | Consolidate and update memory entries |
| `show_lightmem_instance` | Show current instance status |

## Usage Examples

### Via uvx (Recommended)

```bash
export OPENAI_API_KEY="your-api-key"
uvx --from "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light" lightmem-mcp
```

### Local Development

```bash
# Install with MCP dependencies
pip install -e ".[mcp]"

# Run server
export OPENAI_API_KEY="your-api-key"
lightmem-mcp

# Or with config file
lightmem-mcp --config /path/to/config.json
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python -m lightmem.mcp.server
```

### HTTP Transport

```bash
fastmcp run src/lightmem/mcp/server.py:mcp --transport http --port 8000
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│  LightMem MCP    │────▶│   OpenAI API    │
│ (Claude, etc.)  │     │     Server       │     │  (LLM + Embed)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Qdrant (Local)  │
                        │  Vector Storage  │
                        └──────────────────┘
```

- **Pure API mode**: No GPU or local models required
- **OpenAI API**: Used for both LLM operations and embeddings
- **Qdrant**: Local vector database for memory storage
