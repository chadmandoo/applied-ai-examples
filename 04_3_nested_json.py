# =============================================================================
# Nested JSON Parser
# =============================================================================
# Parse LLM output into complex nested JSON structures. Use nested Pydantic
# models to define hierarchical data, and JsonOutputParser will instruct the
# LLM to return properly structured nested JSON.
#
# Key concepts:
#   - Nested Pydantic models: Address inside Company
#   - JsonOutputParser handles nested structures automatically
#   - Access nested data: result['headquarters']['city']
#
# Corresponding endpoint: GET /api/parsers/nested-json
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


# Define nested Pydantic models
# Address will be nested inside Company
class Address(BaseModel):
    """Address information."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class Company(BaseModel):
    """Company with nested address structure."""

    name: str = Field(description="Company name")
    employees: int = Field(description="Number of employees")
    headquarters: Address = Field(description="Company headquarters address")


# Create parser with the nested model
# The format instructions will include the nested structure
parser = JsonOutputParser(pydantic_object=Company)

prompt = ChatPromptTemplate.from_template(
    "Generate a fictional tech company with headquarters address.\n{format_instructions}"
)

formatted_prompt = prompt.format_messages(
    format_instructions=parser.get_format_instructions()
)

# Get LLM response and parse
llm_output = llm.invoke(formatted_prompt).content
company = parser.parse(llm_output)

# Access top-level fields
print(f"Company: {company['name']}")
print(f"Employees: {company['employees']}")

# Access nested fields
print(f"Street: {company['headquarters']['street']}")
print(f"City: {company['headquarters']['city']}")
print(f"Country: {company['headquarters']['country']}")

# Full structure
print(f"\nFull JSON: {company}")

# Example output:
# Company: TechVision Inc
# Employees: 2500
# Street: 123 Innovation Drive
# City: San Francisco
# Country: USA
# Full JSON: {'name': 'TechVision Inc', 'employees': 2500, 'headquarters': {'street': '123 Innovation Drive', 'city': 'San Francisco', 'country': 'USA'}}
