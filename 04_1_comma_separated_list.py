# =============================================================================
# Comma-Separated List Parser
# =============================================================================
# Parse LLM output into a Python list. The parser instructs the LLM to return
# items separated by commas, then automatically splits them into a list.
# Great for simple lists like tags, categories, or keywords.
#
# Key concepts:
#   - CommaSeparatedListOutputParser: Parses comma-separated values
#   - get_format_instructions(): Tells LLM to use comma-separated format
#   - parse(): Splits the string into a Python list
#
# Corresponding endpoint: GET /api/parsers/comma-list
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create the parser - it knows how to split comma-separated values
parser = CommaSeparatedListOutputParser()

# Create a prompt that includes the parser's format instructions
# The format_instructions tell the LLM to return comma-separated values
prompt = ChatPromptTemplate.from_template(
    "List 5 {category}.\n{format_instructions}"
)

# Format the prompt with our category and the parser's instructions
formatted_prompt = prompt.format_messages(
    category="popular programming languages",
    format_instructions=parser.get_format_instructions()
)

# Get the LLM response (will be something like "Python, JavaScript, Java, C++, Go")
llm_output = llm.invoke(formatted_prompt).content

# Parse the comma-separated string into a Python list
parsed_list = parser.parse(llm_output)

print(f"Raw output: {llm_output}")
print(f"Parsed list: {parsed_list}")
print(f"Type: {type(parsed_list)}")
print(f"First item: {parsed_list[0]}")

# Example output:
# Raw output: Python, JavaScript, Java, C++, Go
# Parsed list: ['Python', 'JavaScript', 'Java', 'C++', 'Go']
# Type: <class 'list'>
# First item: Python
