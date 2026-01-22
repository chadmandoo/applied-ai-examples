# Applied AI Examples

Brief code snippets demonstrating LangChain + Ollama patterns.

## Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2
```

## Examples

| File | Description |
|------|-------------|
| `01_basic_invocation.py` | Send a message, get a response |
| `02_system_messages.py` | Add persona/style with SystemMessage |
| `03_usage_metadata.py` | Track input/output tokens |
| `04_prompt_templates.py` | Reusable parameterized prompts |
| `05_json_structured_output.py` | Extract structured data with Pydantic |

## Configuration

```python
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)
```
