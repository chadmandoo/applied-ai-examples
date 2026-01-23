# =============================================================================
# JSON Parser with List Fields
# =============================================================================
# Parse LLM output into JSON containing array/list fields. Use list[str] in
# your Pydantic model to define fields that should contain multiple values.
# Great for tags, categories, related items, etc.
#
# Key concepts:
#   - list[str] type hint: Tells parser to expect an array
#   - JsonOutputParser formats instructions for list fields
#   - Access lists: result['genres'], result['tags']
#
# Corresponding endpoint: GET /api/parsers/json-list
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)


# Define model with list fields
# genres and tags will be arrays in the JSON output
class Book(BaseModel):
    """Book with list fields."""

    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    genres: list[str] = Field(description="List of genres")
    tags: list[str] = Field(description="List of descriptive tags")


# Create parser - it will instruct LLM to return arrays for list fields
parser = JsonOutputParser(pydantic_object=Book)

prompt = ChatPromptTemplate.from_template(
    "Create a fictional science fiction book with multiple genres and tags.\n{format_instructions}"
)

formatted_prompt = prompt.format_messages(
    format_instructions=parser.get_format_instructions()
)

# Get LLM response and parse
llm_output = llm.invoke(formatted_prompt).content
book = parser.parse(llm_output)

# Access scalar fields
print(f"Title: {book['title']}")
print(f"Author: {book['author']}")

# Access list fields
print(f"Genres: {book['genres']}")
print(f"Tags: {book['tags']}")

# Work with the lists
print(f"\nFirst genre: {book['genres'][0]}")
print(f"Number of tags: {len(book['tags'])}")
print(f"Genres joined: {', '.join(book['genres'])}")

# Example output:
# Title: The Quantum Paradox
# Author: Elena Starwright
# Genres: ['Science Fiction', 'Thriller', 'Space Opera']
# Tags: ['time travel', 'AI', 'dystopian', 'adventure']
# First genre: Science Fiction
# Number of tags: 4
# Genres joined: Science Fiction, Thriller, Space Opera
