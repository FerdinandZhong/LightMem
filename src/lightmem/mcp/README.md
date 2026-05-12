# LightMem MCP Server

Launch LightMem as an MCP server via `uvx` for seamless integration with MCP-compatible clients. This branch uses **ChromaDB** as the vector store — no Qdrant required.

## Quick Start

### MCP Client Configuration (Recommended: Remote ChromaDB)

Add to your MCP client config (e.g., Claude Desktop, Cursor, Agent Studio):

```json
{
  "mcpServers": {
    "lightmem": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light-chroma", "lightmem-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "CHROMA_HOST": "${CHROMA_HOST}",
        "LIGHTMEM_COLLECTION_NAME": "${LIGHTMEM_COLLECTION_NAME}"
      }
    }
  }
}
```

> **Why Remote ChromaDB?** Many MCP environments (like Agent Studio) run in sandboxes with filesystem isolation. A remote ChromaDB server ensures data persists across sessions.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | - | OpenAI API key for embeddings and LLM |
| `CHROMA_HOST` | Recommended | - | Remote ChromaDB server host (enables remote mode) |
| `CHROMA_PORT` | No | `8000` | Remote ChromaDB server port |
| `LIGHTMEM_COLLECTION_NAME` | No | `lightmem_memory` | ChromaDB collection name |
| `LIGHTMEM_DATA_PATH` | No | `./lightmem_data` | Local ChromaDB storage path (fallback when `CHROMA_HOST` not set) |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LIGHTMEM_LLM_MODEL` | No | `gpt-4o-mini` | LLM model for memory operations |
| `LIGHTMEM_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `LIGHTMEM_EMBEDDING_DIMS` | No | `1536` | Embedding dimensions (must match the embedding model) |
| `LIGHTMEM_CONFIG_PATH` | No | - | Path to a custom config JSON file |

> **Storage mode**: `CHROMA_HOST` (remote) is **strongly recommended** for production. Use `LIGHTMEM_DATA_PATH` (local) only for development where filesystem persistence is guaranteed.

### Cross-Session Memory Isolation

Use different `LIGHTMEM_COLLECTION_NAME` values to isolate memory between different workflows or users.

### Remote ChromaDB Mode (Recommended)

```json
{
  "mcpServers": {
    "lightmem": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light-chroma", "lightmem-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "CHROMA_HOST": "your-chroma-server",
        "CHROMA_PORT": "8000",
        "LIGHTMEM_COLLECTION_NAME": "my_collection",
        "LIGHTMEM_LLM_MODEL": "gpt-4o-mini",
        "LIGHTMEM_EMBEDDING_MODEL": "text-embedding-3-small",
        "LIGHTMEM_EMBEDDING_DIMS": "1536"
      }
    }
  }
}
```

**Why remote ChromaDB?**
- Guaranteed persistence across sessions
- Works in sandboxed environments (Agent Studio, Docker, etc.)
- Simple to self-host (`docker run -p 8000:8000 chromadb/chroma`)

### Local Storage Mode (Development Only)

For local development where filesystem persistence is guaranteed:

```bash
export LIGHTMEM_DATA_PATH="/path/to/persistent/storage"
```

> **Warning**: Local mode will NOT work in sandboxed environments (Agent Studio, containers with ephemeral filesystems).

### Config File

Alternatively, provide a JSON config file via `LIGHTMEM_CONFIG_PATH` or `--config`. See [example.json](example.json) for a template.

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_timestamp` | Get current timestamp | None |
| `configure_lightmem` | Configure API key, base URL, and model at runtime | `api_key`, `base_url`, `llm_model`, `embedding_model`, `embedding_dims` |
| `add_memory` | Add user/assistant message pair to memory | `user_input`, `assistant_reply`, `timestamp`, `force_segment`, `force_extract` |
| `retrieve_memory` | Query memories by natural language | `query`, `limit`, `filters` |
| `offline_update` | Consolidate and deduplicate memory entries | `top_k`, `keep_top_n`, `score_threshold` |
| `show_lightmem_instance` | Show current instance status | None |

## Usage Examples

### Via uvx (Recommended)

```bash
export OPENAI_API_KEY="your-api-key"
export LIGHTMEM_DATA_PATH="/path/to/persistent/storage"
uvx --from "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light-chroma" lightmem-mcp
```

### Local Development

```bash
# Install dependencies
pip install -e .

# Run server
export OPENAI_API_KEY="your-api-key"
export LIGHTMEM_DATA_PATH="/path/to/persistent/storage"
lightmem-mcp

# Or with a config file
lightmem-mcp --config /path/to/config.json
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uvx --from "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light-chroma" lightmem-mcp
```

### HTTP Transport

```bash
fastmcp run src/lightmem/mcp/server.py:mcp --transport http --port 8000
```

## Architecture

### Remote Mode (`CHROMA_HOST`) — Recommended

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│  LightMem MCP    │────▶│   OpenAI API    │
│ (Claude, etc.)  │     │     Server       │     │  (LLM + Embed)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               │ HTTP
                               ▼
                        ┌──────────────────┐
                        │ ChromaDB (Remote)│
                        │  Vector Server   │
                        └──────────────────┘
```

### Local Mode (`LIGHTMEM_DATA_PATH`) — Development Only

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│  LightMem MCP    │────▶│   OpenAI API    │
│ (Claude, etc.)  │     │     Server       │     │  (LLM + Embed)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  ChromaDB (Local)│
                        │  Vector Storage  │
                        └──────────────────┘
```

- **Pure API mode**: No GPU or local models required
- **OpenAI API**: Used for both LLM operations and embeddings
- **ChromaDB**: Local or remote vector database for memory storage
- **Direct storage**: When `topic_segment` is disabled (default for MCP), memories are stored directly without heavy segmentation dependencies

## Agent Studio Integration

MCP servers in Agent Studio run inside bubblewrap sandboxes with filesystem isolation. **Local storage does not persist** because writes go to a sandboxed virtual filesystem.

### Setup: Deploy ChromaDB + Configure LightMem

1. **Deploy ChromaDB** as a service (e.g., via Docker):

```bash
docker run -d -p 8000:8000 chromadb/chroma
```

2. **Configure LightMem MCP in Agent Studio**:

```json
{
  "env": {
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "CHROMA_HOST": "your-chroma-host",
    "CHROMA_PORT": "8000",
    "LIGHTMEM_COLLECTION_NAME": "my_workflow_memory"
  }
}
```

> **Note**: Local mode (`LIGHTMEM_DATA_PATH`) will NOT work in Agent Studio due to sandbox isolation.
