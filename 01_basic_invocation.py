"""
Example 1: Basic LLM Invocation

This example demonstrates the simplest way to interact with a local LLM using
LangChain and Ollama. It shows how to:
- Create a ChatOllama client
- Send a single message
- Receive and access the response

⚠️ This is a non-compilable example for educational purposes.
"""

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Initialize the Ollama LLM client
# This connects to a locally running Ollama instance
llm = ChatOllama(
    model="llama3.2",                    # Model to use
    base_url="http://localhost:11434",   # Ollama API endpoint
    temperature=0.5                       # Controls randomness (0.0-1.0)
)

# Create a simple message from the user
message = [HumanMessage(content="Say hello world in one sentence.")]

# Invoke the LLM and get the response
response = llm.invoke(message)

# Access the text content of the response
print(response.content)

# Example output:
# "Hello, world - a greeting to the universe and all its wonders!"

"""
Key Takeaways:
--------------
1. ChatOllama is the client for interacting with local Ollama models
2. HumanMessage represents user input
3. invoke() sends messages and returns a response synchronously
4. response.content contains the generated text

Common Parameters:
------------------
- model: The LLM model name (llama3.2, mistral, etc.)
- base_url: Where Ollama is running (default: http://localhost:11434)
- temperature: Creativity level (0.0 = deterministic, 1.0 = very creative)
- num_predict: Max tokens to generate (default: 128)

Use Cases:
----------
- Simple question answering
- Text generation
- Quick prototyping
- Testing LLM connectivity
"""
