# =============================================================================
# Chat Prompt Templates
# =============================================================================
# Structure multi-message prompts with system and human roles. The system
# message configures the AI's behavior, while the human message contains
# the actual query.
#
# Key concepts:
#   - from_messages() creates structured conversation prompts
#   - ("system", "...") sets AI role and behavior
#   - ("human", "...") contains the user's query
#   - Variables work in both system and human messages
#
# Corresponding endpoint: GET /api/prompt-templates/chat
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create a chat prompt with role-based messages
# The tuple format: (role, content)
# - "system": Instructions for the AI
# - "human": The user's input
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who {behavior}. Always respond in a {style} manner."),
    ("human", "{user_input}")
])

# Fill in all variables across both messages
formatted_prompt = chat_prompt.format_messages(
    role="Python programming expert",
    behavior="explains concepts with code examples",
    style="clear and concise",
    user_input="How do I read a CSV file in Python?"
)

# The LLM will adopt the persona defined in the system message
response = llm.invoke(formatted_prompt)

print(response.content)

# Example output: Clear, concise explanation with Python code examples...
