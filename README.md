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

| File | Endpoint | Description |
|------|----------|-------------|
| `01_basic_invocation.py` | `/api/basic` | Send a message, get a response |
| `02_system_messages.py` | `/api/basic2` | Add persona/style with SystemMessage |
| `02_1_usage_metadata.py` | `/api/basic3` | Track input/output tokens |
| `03_prompt_templates.py` | `/api/prompt-templates/basic` | Reusable parameterized prompts |
| `03_1_multi_variable_prompts.py` | `/api/prompt-templates/multi-variable` | Multiple variables in one template |
| `03_2_chat_prompt_templates.py` | `/api/prompt-templates/chat` | Structured system/human messages |
| `03_3_few_shot_prompting.py` | `/api/prompt-templates/few-shot` | Learn patterns from examples |
| `03_4_prompt_composition.py` | `/api/prompt-templates/composition` | Combine reusable prompt components |
| `04_json_structured_output.py` | `/api/parsers/json` | Extract structured data with Pydantic |

## Numbering Scheme

- `01` - Basic invocation
- `02` - System messages & personas
  - `02_1` - Usage metadata/token tracking
- `03` - Prompt templates
  - `03_1` - Multi-variable prompts
  - `03_2` - Chat prompt templates
  - `03_3` - Few-shot prompting
  - `03_4` - Prompt composition
- `04` - Output parsing & structured data

## Configuration

```python
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)
```
