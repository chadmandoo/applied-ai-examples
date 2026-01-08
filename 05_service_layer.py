"""
Example 5: Service Layer Pattern

This example demonstrates the service layer pattern for LLM integration.
Encapsulating LLM logic in a service class provides:
- Clean separation of concerns
- Reusable LLM operations
- Easy testing and mocking
- Consistent error handling
- Better code organization

⚠️ This is a non-compilable example for educational purposes.
"""

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class OllamaService:
    """
    Service class for Ollama LLM operations.

    This class encapsulates all LLM interaction logic, providing a clean
    interface for the rest of the application.
    """

    def __init__(self, llm: ChatOllama):
        """
        Initialize the service with an LLM client.

        Args:
            llm: Configured ChatOllama instance
        """
        self.llm = llm

    def simple_message(self, text: str) -> str:
        """
        Send a simple text message and return the response.

        Args:
            text: User's input text

        Returns:
            LLM's text response

        Example:
            >>> service = OllamaService(llm)
            >>> response = service.simple_message("Say hello")
            >>> print(response)
            "Hello! How can I help you today?"
        """
        message = [HumanMessage(content=text)]
        return self.llm.invoke(message).content

    def simple_message_with_params(self, messages: list) -> str:
        """
        Send a list of messages (system + human) and return the response.

        Args:
            messages: List of SystemMessage and HumanMessage objects

        Returns:
            LLM's text response

        Example:
            >>> messages = [
            ...     SystemMessage(content="You are a poet."),
            ...     HumanMessage(content="Write a haiku")
            ... ]
            >>> response = service.simple_message_with_params(messages)
        """
        return self.llm.invoke(messages).content

    def invoke(self, messages: list):
        """
        Send messages and return the full response object.

        Use this when you need access to metadata like token counts.

        Args:
            messages: List of message objects

        Returns:
            Full response object with content and metadata

        Example:
            >>> response = service.invoke(messages)
            >>> print(response.content)
            >>> print(response.usage_metadata)
        """
        return self.llm.invoke(messages)


# ============================================================================
# Usage Example 1: Simple Message
# ============================================================================

# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.5
)

# Create the service
service = OllamaService(llm)

# Use the service
response = service.simple_message("Say hello world in one sentence.")
print(response)

# ============================================================================
# Usage Example 2: With System Messages
# ============================================================================

from langchain_core.messages import SystemMessage

messages = [
    SystemMessage(content="You are an assistant who responds in the style of Dr Seuss."),
    HumanMessage(content="write me a very short poem about a happy squirrel"),
]

response = service.simple_message_with_params(messages)
print(response)

# ============================================================================
# Usage Example 3: Full Response with Metadata
# ============================================================================

messages = [
    SystemMessage(content="You are an assistant who responds in the style of Shakespeare."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

response = service.invoke(messages)

print(f"Input tokens: {response.usage_metadata.get('input_tokens', 'N/A')}")
print(f"Output tokens: {response.usage_metadata.get('output_tokens', 'N/A')}")
print(f"Content: {response.content}")

"""
Key Takeaways:
--------------
1. Service classes encapsulate LLM logic
2. Constructor injection for dependencies (llm)
3. Multiple methods for different use cases
4. Clear method signatures with type hints
5. Docstrings for documentation

Service Layer Benefits:
-----------------------
1. Separation of Concerns:
   - API layer handles HTTP
   - Service layer handles business logic
   - LLM client handles communication

2. Testability:
   - Easy to mock in tests
   - Can test without real LLM calls
   - Clear dependencies

3. Reusability:
   - Use service in multiple endpoints
   - Share logic across application
   - No code duplication

4. Maintainability:
   - Change LLM logic in one place
   - Clear responsibility boundaries
   - Easy to extend

Advanced Service Pattern:
-------------------------

class AdvancedOllamaService:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.conversation_history = []

    def chat_with_history(self, message: str) -> str:
        \"\"\"Maintain conversation context.\"\"\"
        self.conversation_history.append(HumanMessage(content=message))
        response = self.llm.invoke(self.conversation_history)
        self.conversation_history.append(AIMessage(content=response.content))
        return response.content

    def clear_history(self):
        \"\"\"Clear conversation history.\"\"\"
        self.conversation_history = []

    async def async_invoke(self, messages: list):
        \"\"\"Async version for concurrent requests.\"\"\"
        return await self.llm.ainvoke(messages)

Testing Pattern:
----------------

# test_ollama_service.py
from unittest.mock import Mock

def test_simple_message():
    # Mock the LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Hello!"
    mock_llm.invoke.return_value = mock_response

    # Create service with mock
    service = OllamaService(mock_llm)

    # Test
    result = service.simple_message("Hi")
    assert result == "Hello!"
    mock_llm.invoke.assert_called_once()

Use Cases:
----------
- Production applications
- Clean architecture
- Testable code
- Multiple LLM operations
- Shared business logic
- Easy to extend and maintain
"""
