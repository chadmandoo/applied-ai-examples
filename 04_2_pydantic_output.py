# =============================================================================
# Pydantic Output Parser
# =============================================================================
# Parse LLM output directly into validated Pydantic model instances.
# Unlike JsonOutputParser (which returns dicts), this returns actual Pydantic
# objects with type validation, attribute access, and all Pydantic features.
#
# Key concepts:
#   - PydanticOutputParser: Returns validated Pydantic model instances
#   - Field(ge=0, le=10): Pydantic validation constraints
#   - result.title: Access via attributes (not dict keys)
#   - model_dump(): Convert back to dictionary if needed
#
# Corresponding endpoint: GET /api/parsers/pydantic
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)


# Define a Pydantic model with validation constraints
# Field(ge=0, le=10) ensures rating is between 0 and 10
class Movie(BaseModel):
    """Structured movie information."""

    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    year: int = Field(description="The year the movie was released")
    genre: str = Field(description="The primary genre of the movie")
    rating: float = Field(description="Rating from 0-10", ge=0, le=10)


# Create parser from the Pydantic model
parser = PydanticOutputParser(pydantic_object=Movie)

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_template(
    "Generate information about a famous sci-fi movie.\n{format_instructions}"
)

formatted_prompt = prompt.format_messages(
    format_instructions=parser.get_format_instructions()
)

# Get LLM response and parse into Pydantic object
llm_output = llm.invoke(formatted_prompt).content
movie = parser.parse(llm_output)

# Access data via attributes (not dict keys!)
print(f"Title: {movie.title}")
print(f"Director: {movie.director}")
print(f"Year: {movie.year}")
print(f"Genre: {movie.genre}")
print(f"Rating: {movie.rating}")

# Pydantic features work
print(f"\nType: {type(movie)}")
print(f"As dict: {movie.model_dump()}")

# Example output:
# Title: Blade Runner
# Director: Ridley Scott
# Year: 1982
# Genre: Science Fiction
# Rating: 8.1
# Type: <class '__main__.Movie'>
# As dict: {'title': 'Blade Runner', 'director': 'Ridley Scott', ...}
