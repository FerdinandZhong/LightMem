# /remember - Store Information in Long-Term Memory

Store the current conversation context or specific information in LightMem for future reference.

## Usage

```
/remember [what to remember]
```

## Behavior

When invoked:

1. If specific text is provided after `/remember`, store that information
2. If no text provided, summarize and store the key points from the recent conversation
3. Use the `add_memory` MCP tool to persist the information
4. Confirm what was stored

## Examples

- `/remember` - Store key points from current conversation
- `/remember I prefer dark mode in all applications` - Store specific preference
- `/remember Project deadline is March 15th` - Store specific fact

## Implementation

When this skill is triggered:

1. Extract the information to remember (from argument or conversation summary)
2. Call `add_memory` with:
   - `user_input`: The information or context to remember
   - `assistant_reply`: Acknowledgment of storing the information
   - `force_extract`: true (ensure immediate storage)
3. Respond confirming what was stored
