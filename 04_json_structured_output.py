# =============================================================================
# JSON Structured Output
# =============================================================================
# Extract structured data from unstructured text using Pydantic models and
# JsonOutputParser. This is useful when you need specific fields extracted
# from text (reviews, documents, emails, etc.).
#
# Key concepts:
#   - Pydantic BaseModel: Define the schema for extracted data
#   - Field(): Add descriptions to help the LLM understand each field
#   - JsonOutputParser: Parses LLM output into your defined schema
#   - get_format_instructions(): Generates instructions for the LLM
#
# Corresponding endpoint: GET /api/parsers/json
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


# Define the structure of data we want to extract
# Each field has a type, default value, and description
# The description helps the LLM understand what to look for
class ReviewExtraction(BaseModel):
    """Structured extraction from customer reviews."""

    gift: bool | None = Field(
        default=None,
        description="Was it purchased as a gift?"
    )
    delivery_days: int | None = Field(
        default=None,
        description="Days for delivery"
    )
    price_value: str | None = Field(
        default=None,
        description="Opinion on price vs value"
    )


# Sample text to extract data from
customer_review = """
I bought the Acme 3000 blender as a gift for my mom's birthday, \
and she absolutely loves it! I ordered it online and it arrived in just 3 days. \
It cost more than I expected, but the performance justifies the price.
"""

# Create a parser that knows about our Pydantic model
parser = JsonOutputParser(pydantic_object=ReviewExtraction)

# Define the prompt template with a placeholder for format instructions
# The format_instructions tell the LLM exactly how to structure its output
review_template = """\
Analyze the following customer review and extract the required information. \
If information is not present in the review, use null for that field. \
{format_instructions}
text: ```{customer_review}```
"""

# Create and format the prompt
prompt_template = ChatPromptTemplate.from_template(review_template)
formatted_prompt = prompt_template.format_messages(
    customer_review=customer_review,
    format_instructions=parser.get_format_instructions()  # Auto-generated from Pydantic model
)

# Get the raw LLM output (will be JSON string)
llm_output = llm.invoke(formatted_prompt).content

# Parse the JSON string into a Python dictionary
parsed_output = parser.parse(llm_output)

print(parsed_output)
# Example output: {'gift': True, 'delivery_days': 3, 'price_value': 'expensive but worth it'}
