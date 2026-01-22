# =============================================================================
# Prompt Composition
# =============================================================================
# Build complex prompts from reusable components. Each component handles one
# aspect (system role, context, question), then they're combined into a full
# prompt. Great for maintaining DRY principles in prompt engineering.
#
# Key concepts:
#   - SystemMessagePromptTemplate: Reusable system message component
#   - HumanMessagePromptTemplate: Reusable human message component
#   - from_messages(): Combines components into complete prompt
#   - Components can be mixed and matched as needed
#
# Corresponding endpoint: GET /api/prompt-templates/composition
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create reusable prompt components
# Each component is independent and can be reused across different prompts

# Component 1: System role definition
system_template = SystemMessagePromptTemplate.from_template(
    "You are a {role} with expertise in {domain}."
)

# Component 2: Context provider
context_template = HumanMessagePromptTemplate.from_template(
    """Context: {context}

Please keep this context in mind when answering."""
)

# Component 3: Question template
question_template = HumanMessagePromptTemplate.from_template(
    "Question: {question}"
)

# Compose components into a complete prompt
# Order matters: system → context → question
full_prompt = ChatPromptTemplate.from_messages([
    system_template,
    context_template,
    question_template
])

# Fill in all variables across all components
formatted_prompt = full_prompt.format_messages(
    role="financial advisor",
    domain="retirement planning",
    context="The client is 35 years old, earning $80k/year, with $10k in savings.",
    question="What should they prioritize for retirement planning?"
)

# The LLM receives all components as a coherent conversation
response = llm.invoke(formatted_prompt)

print(response.content)

# Example output: Prioritized retirement planning advice based on the context...
