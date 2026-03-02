# LightMem MCP Server

Launch LightMem as an MCP server via `uvx` for seamless integration with MCP-compatible clients.

## Quick Start

### MCP Client Configuration

Add to your MCP client config (e.g., Claude Desktop, Cursor, Agent Studio):

```json
{
  "mcpServers": {
    "lightmem": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/FerdinandZhong/LightMem.git@mcp-light", "lightmem-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "LIGHTMEM_DATA_PATH": "${LIGHTMEM_DATA_PATH}",
        "LIGHTMEM_COLLECTION_NAME": "${LIGHTMEM_COLLECTION_NAME}"
      }
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | - | OpenAI API key for embeddings and LLM |
| `LIGHTMEM_DATA_PATH` | **Yes*** | `./lightmem_data` | Path for Qdrant vector storage |
| `LIGHTMEM_COLLECTION_NAME` | No | `lightmem_memory` | Qdrant collection name |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `LIGHTMEM_LLM_MODEL` | No | `gpt-4o-mini` | LLM model for memory operations |
| `LIGHTMEM_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `LIGHTMEM_EMBEDDING_DIMS` | No | `1536` | Embedding dimensions |
| `LIGHTMEM_CONFIG_PATH` | No | - | Path to custom config JSON file |

> **\*Required for Cross-Session Memory**: While `LIGHTMEM_DATA_PATH` has a default value, you **must** set it explicitly to a persistent, shared location for memories to persist across sessions.

### Cross-Session Memory Setup

For memories to persist across conversations/sessions, `LIGHTMEM_DATA_PATH` must point to a **shared, persistent location**:

```bash
# Good - Persistent paths
export LIGHTMEM_DATA_PATH="/data/shared/lightmem/"
export LIGHTMEM_DATA_PATH="/home/cdsw/lightmem_data/"

# Bad - Session-specific paths (memory will be lost!)
# workflows/.../session/abc123/lightmem_data
```

**Use different `LIGHTMEM_COLLECTION_NAME` values** to isolate data between different workflows or use cases.

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
- **Direct Storage**: When `topic_segment` is disabled (default for MCP), memories are stored directly without heavy segmentation dependencies

## Agent Studio Integration

When using LightMem MCP in Cloudera Agent Studio:

1. **Set persistent storage path** via `LIGHTMEM_DATA_PATH` environment variable
2. **Avoid session-specific folders** - these are cleared between conversations
3. **Use unique collection names** to separate data between workflows

Example workflow variables:
```
LIGHTMEM_DATA_PATH = /data/shared/lightmem/my_workflow/
LIGHTMEM_COLLECTION_NAME = my_workflow_memory
```
