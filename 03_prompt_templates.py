# =============================================================================
# Prompt Templates
# =============================================================================
# Create reusable, parameterized prompts using ChatPromptTemplate.
# Instead of hardcoding prompts, use variables that can be filled in at runtime.
#
# Key concepts:
#   - ChatPromptTemplate: Factory for creating formatted prompts
#   - {variable} syntax: Placeholders that get replaced with actual values
#   - format_messages(): Fills in the variables and returns message objects
#
# Corresponding endpoint: GET /api/prompt-templates/basic
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Define a template string with placeholders
# {style} and {text} will be replaced with actual values
# Using triple backticks to clearly delimit the input text
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```

Return only the translated text.
"""

# Create a prompt template from the string
prompt_template = ChatPromptTemplate.from_template(template_string)

# Fill in the template variables to create actual messages
# This returns a list of message objects ready for the LLM
formatted_prompt = prompt_template.format_messages(
    style="slang",
    text="Hello how art thou?"
)

# Send the formatted prompt to the LLM
response = llm.invoke(formatted_prompt)

# Output will be the text translated into the requested style
print(response.content)

# Example output: "Yo, what's good?"
