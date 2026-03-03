# /recall - Retrieve Information from Long-Term Memory

Search and retrieve previously stored information from LightMem.

## Usage

```
/recall <search query>
```

## Behavior

When invoked:

1. Use the provided search query to find relevant memories
2. Call the `retrieve_memory` MCP tool
3. Present the retrieved information in a clear format
4. Offer to provide more details if needed

## Examples

- `/recall project deadlines` - Find stored project deadline information
- `/recall user preferences` - Retrieve stored user preferences
- `/recall last week's discussion` - Find memories from recent conversations
- `/recall Python code style` - Search for coding preferences

## Implementation

When this skill is triggered:

1. Parse the search query from the argument
2. Call `retrieve_memory` with:
   - `query`: The search query provided
   - `limit`: 10 (default, adjust based on context)
3. Format and present the results
4. If no results found, suggest alternative queries or offer to store new information

## Output Format

Present retrieved memories as:
- Summary of what was found
- Key details from each relevant memory
- Timestamps for context
- Offer follow-up actions (more details, store new info, etc.)
