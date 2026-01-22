# =============================================================================
# System Messages & Persona
# =============================================================================
# Add personality, context, or instructions to your LLM using system messages.
# System messages tell the LLM HOW to behave, while human messages tell it
# WHAT to do.
#
# Key concepts:
#   - SystemMessage: Instructions/persona for the LLM (processed first)
#   - HumanMessage: The actual user query
#   - Message order matters: System message should come before human message
#
# Corresponding endpoint: GET /api/basic2
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create a message list with both system and human messages
# The SystemMessage sets the persona/style for all responses
# The HumanMessage contains the actual request
messages = [
    SystemMessage(content="You are an assistant who responds in the style of Dr Seuss."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

# The LLM will follow the system message instructions when generating the response
response = llm.invoke(messages)

# The output will be in Dr. Seuss style due to the system message
print(response.content)
