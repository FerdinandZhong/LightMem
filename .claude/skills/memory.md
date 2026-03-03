# LightMem Memory Management

Manage long-term memory using the LightMem MCP server. This skill helps you store, retrieve, and manage persistent memories across conversations.

## Available MCP Tools

When the LightMem MCP server is configured, you have access to these tools:

### `add_memory`
Store a conversation exchange in long-term memory.
- `user_input` (required): The user's message or question
- `assistant_reply` (required): The assistant's response
- `timestamp` (optional): ISO timestamp, defaults to current time
- `force_extract` (optional): Force memory extraction regardless of thresholds

### `retrieve_memory`
Query stored memories by natural language.
- `query` (required): Natural language search query
- `limit` (optional): Number of results to return (default: 10)
- `filters` (optional): Metadata filters for the search

### `offline_update`
Consolidate and deduplicate memory entries.
- `top_k` (optional): Neighbors to consider (default: 20)
- `keep_top_n` (optional): Top entries to keep (default: 10)
- `score_threshold` (optional): Similarity threshold (default: 0.8)

### `get_timestamp`
Get the current timestamp in ISO format.

### `show_lightmem_instance`
Display the current LightMem configuration and status.

## When to Use Memory

### Store memories when:
- User shares personal preferences, facts, or important information
- User explicitly asks to "remember this" or "save this"
- A decision or agreement is made that should persist
- User provides context about their work, projects, or goals

### Retrieve memories when:
- User asks about something discussed previously
- User references past conversations
- Context from previous interactions would help the current task
- User asks "what do you know about..." or "what did I tell you about..."

## Usage Patterns

### Remembering Information
When user says things like "remember that I prefer X" or shares important facts:
1. Acknowledge the information
2. Call `add_memory` with the exchange
3. Confirm it's been stored

### Recalling Information
When user asks about past discussions:
1. Call `retrieve_memory` with a relevant query
2. Use the retrieved context in your response
3. Cite the memory source if relevant

### Proactive Memory Use
Before answering questions that might benefit from context:
1. Consider if past conversations are relevant
2. Query memory for related information
3. Incorporate relevant context into your response

## Example Interactions

**User**: "Remember that my favorite programming language is Rust"
**Action**: Call `add_memory(user_input="Remember that my favorite programming language is Rust", assistant_reply="I'll remember that Rust is your favorite programming language.")`

**User**: "What's my preferred coding style?"
**Action**: Call `retrieve_memory(query="coding style preferences")` then respond based on results

**User**: "Can you help me with a project? It's the same one we discussed last week."
**Action**: Call `retrieve_memory(query="project discussed last week")` to get context before helping

## Best Practices

1. **Be selective**: Don't store every interaction, focus on meaningful information
2. **Use descriptive queries**: More specific queries yield better retrieval results
3. **Consolidate periodically**: Run `offline_update` to merge similar memories
4. **Respect privacy**: Don't store sensitive information without user consent
5. **Verify before acting**: When retrieving memories, verify with user if context is correct
