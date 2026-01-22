# =============================================================================
# Usage Metadata & Token Tracking
# =============================================================================
# Monitor LLM usage by tracking input and output tokens. This is essential for:
#   - Cost estimation (API providers charge per token)
#   - Performance monitoring (longer responses = more latency)
#   - Rate limiting and quota management
#
# Key concepts:
#   - usage_metadata: Dictionary attached to every response
#   - input_tokens: How many tokens were in your prompt
#   - output_tokens: How many tokens the LLM generated
#
# Corresponding endpoint: GET /api/basic3
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create messages with a system persona
messages = [
    SystemMessage(content="You are an assistant who responds in the style of Shakespeare."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

# Invoke the LLM
response = llm.invoke(messages)

# Access token usage from the response's usage_metadata dictionary
# Use .get() with a default value in case the field is missing
input_tokens = response.usage_metadata.get('input_tokens', 'N/A')
output_tokens = response.usage_metadata.get('output_tokens', 'N/A')

print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Content: {response.content}")

# Example output:
# Input tokens: 34
# Output tokens: 89
# Content: Hark! A squirrel doth frolic with glee...
