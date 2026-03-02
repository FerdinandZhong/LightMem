# LightMem MCP Server

Launch LightMem as an MCP server via `uvx` for seamless integration with MCP-compatible clients.

## Quick Start

### MCP Client Configuration (Recommended: Remote Qdrant)

Add to your MCP client config (e.g., Claude Desktop, Cursor, Agent Studio):

```json
{
  "mcpServers": {
    "lightmem": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light", "lightmem-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "QDRANT_URL": "${QDRANT_URL}",
        "LIGHTMEM_COLLECTION_NAME": "${LIGHTMEM_COLLECTION_NAME}"
      }
    }
  }
}
```

> **Why Remote Qdrant?** Many MCP environments (like Agent Studio) run in sandboxes with filesystem isolation. Remote Qdrant ensures data persistence across sessions.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | - | OpenAI API key for embeddings and LLM |
| `QDRANT_URL` | **Recommended** | - | Remote Qdrant server URL (recommended for persistence) |
| `QDRANT_API_KEY` | No | - | API key for remote Qdrant server (if required) |
| `LIGHTMEM_COLLECTION_NAME` | No | `lightmem_memory` | Qdrant collection name |
| `LIGHTMEM_DATA_PATH` | No* | `./lightmem_data` | Path for local Qdrant storage (fallback) |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LIGHTMEM_LLM_MODEL` | No | `gpt-4o-mini` | LLM model for memory operations |
| `LIGHTMEM_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `LIGHTMEM_EMBEDDING_DIMS` | No | `1536` | Embedding dimensions |
| `LIGHTMEM_CONFIG_PATH` | No | - | Path to custom config JSON file |

> **\*Storage Mode**: `QDRANT_URL` (remote) is **strongly recommended** for production use. Use `LIGHTMEM_DATA_PATH` (local) only for development/testing where filesystem persistence is guaranteed.

### Cross-Session Memory Setup

**Use different `LIGHTMEM_COLLECTION_NAME` values** to isolate data between different workflows or use cases.

### Remote Qdrant Mode (Default & Recommended)

Remote Qdrant is the **recommended approach** for all production deployments:

```json
{
  "mcpServers": {
    "lightmem": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light", "lightmem-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "QDRANT_URL": "https://your-qdrant-server:6333",
        "QDRANT_API_KEY": "${QDRANT_API_KEY}",
        "LIGHTMEM_COLLECTION_NAME": "my_collection"
      }
    }
  }
}
```

**Why Remote Qdrant?**
- Guaranteed data persistence across sessions
- Works in sandboxed environments (Agent Studio, Docker, etc.)
- Scales better for multi-agent or multi-user scenarios
- Easier to backup and manage

When `QDRANT_URL` is set, LightMem connects to the remote Qdrant server instead of using local storage.

### Local Storage Mode (Development Only)

For local development/testing where filesystem persistence is guaranteed:

```bash
export LIGHTMEM_DATA_PATH="/path/to/persistent/storage"
```

> **Warning**: Local mode will NOT work in sandboxed environments (Agent Studio MCP servers, containers with ephemeral filesystems). Use remote Qdrant for these cases.

### Config File

Alternatively, provide a JSON config file via `LIGHTMEM_CONFIG_PATH` or `--config` flag. See [example.json](example.json) for the API-only configuration template.

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_timestamp` | Get current timestamp | None |
| `add_memory` | Add user/assistant message pair to memory | `user_input`, `assistant_reply`, `timestamp`, `force_segment`, `force_extract` |
| `retrieve_memory` | Query memories by natural language | `query`, `limit`, `filters` |
| `offline_update` | Consolidate and update memory entries | `top_k`, `keep_top_n`, `score_threshold` |
| `show_lightmem_instance` | Show current instance status | None |

## Usage Examples

### Via uvx (Recommended)

```bash
export OPENAI_API_KEY="your-api-key"
export LIGHTMEM_DATA_PATH="/path/to/persistent/storage"
uvx --from "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light" lightmem-mcp
```

### Local Development

```bash
# Install with MCP dependencies
pip install -e ".[mcp]"

# Run server
export OPENAI_API_KEY="your-api-key"
export LIGHTMEM_DATA_PATH="/path/to/persistent/storage"
lightmem-mcp

# Or with config file
lightmem-mcp --config /path/to/config.json
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uvx --from "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light" lightmem-mcp
```

### HTTP Transport

```bash
fastmcp run src/lightmem/mcp/server.py:mcp --transport http --port 8000
```

## Architecture

### Remote Mode (QDRANT_URL) - Default & Recommended

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│  LightMem MCP    │────▶│   OpenAI API    │
│ (Claude, etc.)  │     │     Server       │     │  (LLM + Embed)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               │ HTTP
                               ▼
                        ┌──────────────────┐
                        │ Qdrant (Remote)  │
                        │  Vector Server   │
                        └──────────────────┘
```

### Local Mode (LIGHTMEM_DATA_PATH) - Development Only

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
- **Qdrant**: Local or remote vector database for memory storage
- **Direct Storage**: When `topic_segment` is disabled (default for MCP), memories are stored directly without heavy segmentation dependencies

## Agent Studio Integration

MCP servers in Agent Studio run inside bubblewrap sandboxes with filesystem isolation. **Local storage does not persist** because writes go to a sandboxed virtual filesystem.

### Setup: Deploy Qdrant + Configure LightMem

1. **Deploy Qdrant as a CAI Application** (see [SP_hol/qdrant_cai_app](https://github.com/cloudera/SP_hol/tree/main/qdrant_cai_app))

2. **Configure LightMem MCP in Agent Studio**:

```json
{
  "env": {
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "QDRANT_URL": "https://your-cai-domain/qdrant-projectid",
    "LIGHTMEM_COLLECTION_NAME": "my_workflow_memory"
  }
}
```

> **Note**: Local mode (`LIGHTMEM_DATA_PATH`) will NOT work in Agent Studio due to sandbox isolation.
