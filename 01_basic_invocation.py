# =============================================================================
# Basic LLM Invocation
# =============================================================================
# The simplest pattern: send a message to an LLM and get a response.
#
# Key concepts:
#   - ChatOllama: LangChain's client for Ollama local LLM runtime
#   - HumanMessage: Represents user input in the conversation
#   - invoke(): Sends messages to the LLM and returns the response
#
# Corresponding endpoint: GET /api/basic
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Initialize the Ollama client
# - base_url: Where Ollama is running (default localhost:11434)
# - model: Which model to use (must be pulled first via `ollama pull llama3.2`)
# - temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create a list of messages to send
# HumanMessage represents what the user is asking
messages = [HumanMessage(content="Say hello world in one sentence.")]

# Send the messages to the LLM and get a response
# The response object contains .content (the text) and metadata
response = llm.invoke(messages)

# Access the generated text
print(response.content)
