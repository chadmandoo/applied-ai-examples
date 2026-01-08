# Applied AI Examples

This directory contains educational code snippets demonstrating LangChain and LLM integration patterns extracted from the chadpeppers.dev backend project.

**⚠️ Important**: These are non-compilable code examples for educational purposes. They illustrate patterns and approaches but are not meant to be run directly.

## Overview

These examples demonstrate how to integrate Large Language Models (LLMs) using LangChain with local models via Ollama. Each example builds upon the previous one, showing progressively more advanced patterns.

## Technology Stack

- **LangChain**: Framework for building LLM applications
- **langchain-ollama**: Integration with Ollama for local LLM inference
- **Ollama**: Local LLM runtime (runs models like llama3.2 locally)

## Prerequisites

To understand these examples, you should have:
- Python 3.14+ knowledge
- Basic understanding of async/await
- Familiarity with REST APIs
- Ollama installed and running locally

### Ollama Setup

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Verify Ollama is running
curl http://localhost:11434/api/version
```

## Examples

### 1. Basic LLM Invocation (`01_basic_invocation.py`)

The simplest pattern: send a message to an LLM and get a response.

**Key Concepts:**
- Creating a `ChatOllama` client
- Using `HumanMessage` for user input
- Simple `invoke()` call
- Accessing response content

**Use Cases:**
- Simple chatbot queries
- One-off text generation
- Quick LLM testing

---

### 2. System Messages & Persona (`02_system_messages.py`)

Add personality and context to your LLM using system messages.

**Key Concepts:**
- `SystemMessage` for LLM instructions
- `HumanMessage` for user queries
- Message array ordering matters
- Persona/style customization

**Use Cases:**
- Role-playing chatbots
- Specialized assistants (code reviewer, creative writer)
- Consistent tone/style enforcement

---

### 3. Usage Metadata & Token Tracking (`03_usage_metadata.py`)

Monitor LLM usage for cost tracking and optimization.

**Key Concepts:**
- Accessing `usage_metadata` from responses
- Tracking input/output tokens
- Cost estimation
- Performance monitoring

**Use Cases:**
- Production cost tracking
- Rate limiting
- Performance optimization
- Budget management

---

### 4. Prompt Templates (`04_prompt_templates.py`)

Create reusable, parameterized prompts for consistent LLM interactions.

**Key Concepts:**
- `ChatPromptTemplate` for reusable prompts
- Variable substitution with `{variable}` syntax
- Separating prompt logic from application code
- Formatted message generation

**Use Cases:**
- Translation services
- Content transformation
- Standardized responses
- Multi-language support

---

### 5. Service Layer Pattern (`05_service_layer.py`)

Production-ready pattern: encapsulate LLM logic in a service class.

**Key Concepts:**
- Class-based service design
- Dependency injection
- Separation of concerns
- Reusable LLM client

**Use Cases:**
- Production applications
- Testable LLM integration
- Multiple LLM operations
- Clean architecture

---

### 6. Dependency Injection with FastAPI (`06_dependency_injection.py`)

Integrate LLM services with FastAPI using dependency injection.

**Key Concepts:**
- `@lru_cache` for singleton pattern
- FastAPI `Depends()` for DI
- Efficient resource management
- Testable API endpoints

**Use Cases:**
- REST API development
- Microservices
- Scalable web applications
- Testing and mocking

---

## Pattern Progression

The examples follow this learning path:

1. **Simple**: Direct LLM calls
2. **Contextual**: Add system messages for persona
3. **Observable**: Track usage and costs
4. **Templated**: Reusable prompt patterns
5. **Structured**: Service layer abstraction
6. **Production**: Full FastAPI integration

## Configuration

All examples assume Ollama is configured with these defaults:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
TEMPERATURE = 0.5
```

You can customize by:
- Changing the model (e.g., `llama2`, `mistral`)
- Adjusting temperature (0.0 = deterministic, 1.0 = creative)
- Using different base URLs for remote Ollama instances

## Common Patterns

### Error Handling

```python
try:
    response = llm.invoke(messages)
except Exception as e:
    # Handle Ollama connection errors
    raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")
```

### Message Structure

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="User's question here")
]
```

### Response Access

```python
response = llm.invoke(messages)

# Get text content
text = response.content

# Get usage metadata
input_tokens = response.usage_metadata.get('input_tokens', 0)
output_tokens = response.usage_metadata.get('output_tokens', 0)
```

## Next Steps

After understanding these examples, you can:

1. **Explore Advanced LangChain Features:**
   - Chains (sequential LLM operations)
   - Agents (autonomous LLM decision-making)
   - Memory (conversation history)
   - RAG (Retrieval-Augmented Generation)

2. **Production Considerations:**
   - Rate limiting
   - Caching responses
   - Streaming responses
   - Error recovery
   - Monitoring and logging

3. **Alternative LLM Providers:**
   - OpenAI (GPT-4, GPT-3.5)
   - Anthropic (Claude)
   - Google (Gemini)
   - Azure OpenAI

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)

## License

These examples are provided for educational purposes. See the main project LICENSE for details.

## Contributing

These examples are extracted from the production codebase at chadpeppers.dev. For the full, working implementation, see the main project repository.
