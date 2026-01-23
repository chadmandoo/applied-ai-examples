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
| `04_1_comma_separated_list.py` | `/api/parsers/comma-list` | Parse output into Python list |
| `04_2_pydantic_output.py` | `/api/parsers/pydantic` | Validated Pydantic model instances |
| `04_3_nested_json.py` | `/api/parsers/nested-json` | Nested/hierarchical JSON structures |
| `04_4_json_with_lists.py` | `/api/parsers/json-list` | JSON with array fields |
| `05_chains.py` | `/api/chains/basic` | Basic chain composition with LCEL |
| `05_1_sequential_chains.py` | `/api/chains/sequential` | Output of one chain feeds the next |
| `06_runnable.py` | `/api/runnable/passthrough` | RunnablePassthrough for data flow |
| `06_1_parallel_runnable.py` | `/api/runnable/parallel` | Run multiple chains concurrently |
| `07_memory.py` | `/api/memory/continue` | SQLite-based conversation memory |
| `08_tools.py` | `/api/tools/basic` | Create tools with @tool decorator |
| `09_functions.py` | `/api/functions/intent-parsing` | LLM function calling and routing |
| `10_agents.py` | `/api/agents/react-simple` | ReAct pattern: Reasoning + Acting |
| `11_routing.py` | `/api/routing/classify` | Dynamic chain selection |
| `12_document_retrieval.py` | `/api/retrieval/rag` | RAG pattern (stubbed for vector DB) |
| `13_langgraph.py` | `/api/langgraph/basic-flow` | Stateful workflows with SQLite |

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
  - `04_1` - Comma-separated list parser
  - `04_2` - Pydantic output parser
  - `04_3` - Nested JSON structures
  - `04_4` - JSON with list fields
- `05` - Chains
  - `05_1` - Sequential chains
- `06` - Runnable interface (LCEL)
  - `06_1` - Parallel execution
- `07` - Memory (SQLite-based)
- `08` - Tools
- `09` - Functions
- `10` - Agents (ReAct pattern)
- `11` - Routing
- `12` - Document Retrieval (RAG) *stubbed*
- `13` - LangGraph (stateful workflows)

## Configuration

```python
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)
```

## Notes

- **Memory (07)**: Uses SQLite for persistence, reads only from pre-seeded data
- **Document Retrieval (12)**: Stubbed with mock data. Connect Milvus or Pinecone for production
- **LangGraph (13)**: Uses SQLite for workflow state persistence
