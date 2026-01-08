"""
Example 6: Dependency Injection with FastAPI

This example demonstrates the complete production pattern using FastAPI's
dependency injection system. This pattern provides:
- Efficient resource management (singleton pattern)
- Easy testing and mocking
- Clean API endpoint code
- Automatic dependency resolution

⚠️ This is a non-compilable example for educational purposes.
"""

from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# ============================================================================
# Configuration
# ============================================================================

class Settings:
    """Application settings (typically loaded from environment variables)."""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

settings = Settings()


# ============================================================================
# Service Layer
# ============================================================================

class OllamaService:
    """Service class for Ollama LLM operations."""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def simple_message(self, text: str) -> str:
        """Send a simple message and return the response."""
        message = [HumanMessage(content=text)]
        return self.llm.invoke(message).content

    def simple_message_with_params(self, messages: list) -> str:
        """Send messages with system context."""
        return self.llm.invoke(messages).content

    def invoke(self, messages):
        """Return full response object with metadata."""
        return self.llm.invoke(messages)


# ============================================================================
# Dependency Injection
# ============================================================================

@lru_cache(maxsize=1)
def get_ollama_llm() -> ChatOllama:
    """
    Create and cache the Ollama LLM client.

    The @lru_cache decorator ensures this is created once per process
    and reused for all requests. This is efficient because:
    - Avoids creating new connections for each request
    - Shares the LLM client across all endpoints
    - Reduces initialization overhead

    Returns:
        Configured ChatOllama instance
    """
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.5
    )
    return llm


def get_ollama_service(llm: ChatOllama = Depends(get_ollama_llm)) -> OllamaService:
    """
    Create an OllamaService with dependency injection.

    FastAPI will automatically:
    1. Call get_ollama_llm() to get the LLM client
    2. Pass it to this function
    3. Return the service to the endpoint

    Args:
        llm: ChatOllama instance (injected by FastAPI)

    Returns:
        Configured OllamaService instance
    """
    return OllamaService(llm)


# ============================================================================
# API Endpoints
# ============================================================================

router = APIRouter(prefix="/api", tags=["basic"])


@router.get("/basic")
def basic(svc: OllamaService = Depends(get_ollama_service)):
    """
    Simple LLM endpoint.

    The service is injected automatically by FastAPI via Depends().
    """
    try:
        return {"response": svc.simple_message("Say hello world in one sentence.")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


@router.get("/basic2")
def basic2(svc: OllamaService = Depends(get_ollama_service)):
    """
    LLM endpoint with system message (Dr. Seuss persona).
    """
    try:
        messages = [
            SystemMessage(content="You are an assistant who responds in the style of Dr Seuss."),
            HumanMessage(content="write me a very short poem about a happy squirrel"),
        ]
        return {"response": svc.simple_message_with_params(messages)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


@router.get("/basic3")
def basic3(svc: OllamaService = Depends(get_ollama_service)):
    """
    LLM endpoint with usage metadata tracking.
    """
    try:
        messages = [
            SystemMessage(content="You are an assistant who responds in the style of Shakespeare."),
            HumanMessage(content="Write me a very short poem about a happy squirrel"),
        ]

        response = svc.invoke(messages)

        return {
            "response": {
                "Input tokens:": response.usage_metadata.get('input_tokens', 'N/A'),
                "Output tokens:": response.usage_metadata.get('output_tokens', 'N/A'),
                "Content": response.content
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


@router.get("/basic4")
def basic4(svc: OllamaService = Depends(get_ollama_service)):
    """
    LLM endpoint using prompt templates.
    """
    try:
        template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```

Return only the translated text.
"""

        prompt_template = ChatPromptTemplate.from_template(template_string)

        formatted_prompt = prompt_template.format_messages(
            style="slang",
            text="Hello how art thou?"
        )

        return {"response": svc.invoke(formatted_prompt).content}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


"""
Key Takeaways:
--------------
1. @lru_cache creates singleton instances
2. Depends() handles automatic dependency injection
3. Service layer keeps endpoints clean
4. Easy to test by mocking dependencies

Dependency Injection Benefits:
------------------------------
1. Efficiency:
   - LLM client created once, reused forever
   - No connection overhead per request
   - Shared resources across endpoints

2. Testability:
   - Override dependencies in tests
   - Mock LLM without changing code
   - Test endpoints independently

3. Flexibility:
   - Swap implementations easily
   - Different configs for dev/prod
   - Easy to add caching, logging, etc.

4. Clean Code:
   - Endpoints focus on business logic
   - No initialization boilerplate
   - Clear dependency chain

Testing with Dependency Injection:
----------------------------------

# test_api.py
from fastapi.testclient import TestClient
from unittest.mock import Mock

def test_basic_endpoint():
    # Create mock service
    mock_service = Mock()
    mock_service.simple_message.return_value = "Hello!"

    # Override dependency
    app.dependency_overrides[get_ollama_service] = lambda: mock_service

    # Test
    client = TestClient(app)
    response = client.get("/api/basic")

    assert response.status_code == 200
    assert response.json() == {"response": "Hello!"}
    mock_service.simple_message.assert_called_once()

Advanced Patterns:
------------------

# Multiple LLM providers
def get_llm_service(provider: str = "ollama") -> BaseLLMService:
    if provider == "ollama":
        return get_ollama_service()
    elif provider == "openai":
        return get_openai_service()
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Request-scoped dependencies (not cached)
def get_request_context() -> RequestContext:
    return RequestContext(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc)
    )

# Nested dependencies
def get_advanced_service(
    llm: ChatOllama = Depends(get_ollama_llm),
    cache: Cache = Depends(get_cache),
    logger: Logger = Depends(get_logger)
) -> AdvancedService:
    return AdvancedService(llm, cache, logger)

Production Considerations:
--------------------------
1. Add rate limiting per user/IP
2. Implement response caching
3. Add request/response logging
4. Monitor token usage per endpoint
5. Implement circuit breakers
6. Add health checks for LLM availability
7. Use connection pooling for scaling

Use Cases:
----------
- Production REST APIs
- Microservices architecture
- Testable web applications
- Scalable LLM services
- Multi-tenant applications
"""
