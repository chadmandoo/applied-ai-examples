"""
Example 7: JSON Structured Output

This example demonstrates how to use JsonOutputParser with Pydantic models
to extract structured data from LLM responses. This is essential for:
- Extracting specific fields from unstructured text
- Getting consistent, parseable responses
- Building reliable data pipelines with LLMs
- Integrating LLM output with typed systems

⚠️ This is a non-compilable example for educational purposes.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Initialize the LLM client
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)

# ============================================================================
# Step 1: Define a Pydantic Model for Structured Output
# ============================================================================

# The Pydantic model defines the schema the LLM should follow.
# Field descriptions help the LLM understand what each field should contain.
# Use Optional fields (| None) for data that may not be present in the input.

class ReviewExtraction(BaseModel):
    """Structured extraction from customer reviews."""
    gift: bool | None = Field(
        default=None,
        description="Indicates if the review mentions giving the product as a gift, null if not mentioned"
    )
    delivery_days: int | None = Field(
        default=None,
        description="Number of days taken for delivery as mentioned in the review, null if not mentioned"
    )
    price_value: str | None = Field(
        default=None,
        description="Customer's opinion on the price vs value of the product, null if not mentioned"
    )

# ============================================================================
# Step 2: Create the JsonOutputParser
# ============================================================================

# The parser generates format instructions and parses the LLM output
parser = JsonOutputParser(pydantic_object=ReviewExtraction)

# Get format instructions to include in the prompt
# This tells the LLM exactly how to structure its response
format_instructions = parser.get_format_instructions()

print("Format Instructions:")
print(format_instructions)
# Output:
# The output should be formatted as a JSON instance that conforms to the
# JSON schema below...

# ============================================================================
# Step 3: Create the Prompt Template
# ============================================================================

# The review text to analyze
customer_review = """
I recently purchased the Acme 3000 blender and I have to say, \
it has completely transformed my kitchen experience. \
The powerful motor crushes ice and blends fruits effortlessly, \
making my morning smoothies a breeze. \
The sleek design also looks great on my countertop. \
Cleanup is simple too, as the parts are dishwasher safe. \
Overall, I'm extremely satisfied with my purchase and would highly recommend \
the Acme 3000 to anyone in need of a reliable and efficient blender. It cost more \
than I expected, but the performance justifies the price.
"""

# Include instructions to use null for missing information
# This prevents the LLM from hallucinating values
review_template = """\
Analyze the following customer review and extract the required information. \
If information is not present in the review, use null for that field. \
{format_instructions}
text: ```{customer_review}```
"""

prompt_template = ChatPromptTemplate.from_template(review_template)

# Format the prompt with the review and format instructions
formatted_prompt = prompt_template.format_messages(
    customer_review=customer_review,
    format_instructions=format_instructions
)

# ============================================================================
# Step 4: Invoke LLM and Parse Response
# ============================================================================

# Get the raw LLM response
llm_output = llm.invoke(formatted_prompt).content
print("Raw LLM Output:")
print(llm_output)
# Output:
# {"gift": null, "delivery_days": null, "price_value": "Cost more than expected but performance justifies the price"}

# Parse the response into a structured dictionary
parsed_output = parser.parse(llm_output)
print("Parsed Output:")
print(parsed_output)
# Output:
# {'gift': None, 'delivery_days': None, 'price_value': 'Cost more than expected but performance justifies the price'}

# Access individual fields with type safety
print(f"Gift mentioned: {parsed_output.get('gift')}")
print(f"Delivery days: {parsed_output.get('delivery_days')}")
print(f"Price value: {parsed_output.get('price_value')}")

# ============================================================================
# Example 7b: Different Schema for Different Use Cases
# ============================================================================

class SentimentAnalysis(BaseModel):
    """Sentiment analysis extraction."""
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_phrases: list[str] = Field(description="List of key phrases that influenced the sentiment")

sentiment_parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

sentiment_template = """\
Analyze the sentiment of the following text.
{format_instructions}
text: ```{text}```
"""

sentiment_prompt = ChatPromptTemplate.from_template(sentiment_template)
formatted = sentiment_prompt.format_messages(
    text="This product exceeded all my expectations! Absolutely love it.",
    format_instructions=sentiment_parser.get_format_instructions()
)

response = llm.invoke(formatted)
sentiment_result = sentiment_parser.parse(response.content)
print(sentiment_result)
# Output:
# {'sentiment': 'positive', 'confidence': 0.95, 'key_phrases': ['exceeded expectations', 'absolutely love']}

# ============================================================================
# Example 7c: Nested Structures
# ============================================================================

class Address(BaseModel):
    """Address components."""
    street: str | None = Field(default=None, description="Street address")
    city: str | None = Field(default=None, description="City name")
    state: str | None = Field(default=None, description="State or province")
    zip_code: str | None = Field(default=None, description="Postal/ZIP code")

class ContactExtraction(BaseModel):
    """Extract contact information from text."""
    name: str | None = Field(default=None, description="Person's full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    address: Address | None = Field(default=None, description="Physical address if mentioned")

contact_parser = JsonOutputParser(pydantic_object=ContactExtraction)

contact_text = """
Please contact John Smith at john.smith@example.com or call 555-123-4567.
He works at 123 Main Street, Springfield, IL 62701.
"""

contact_template = """\
Extract all contact information from the text.
If information is not present, use null.
{format_instructions}
text: ```{text}```
"""

contact_prompt = ChatPromptTemplate.from_template(contact_template)
formatted = contact_prompt.format_messages(
    text=contact_text,
    format_instructions=contact_parser.get_format_instructions()
)

response = llm.invoke(formatted)
contact_result = contact_parser.parse(response.content)
print(contact_result)
# Output:
# {
#   'name': 'John Smith',
#   'email': 'john.smith@example.com',
#   'phone': '555-123-4567',
#   'address': {
#     'street': '123 Main Street',
#     'city': 'Springfield',
#     'state': 'IL',
#     'zip_code': '62701'
#   }
# }

"""
Key Takeaways:
--------------
1. JsonOutputParser + Pydantic = reliable structured output
2. Field descriptions guide the LLM's extraction
3. Optional fields (| None) prevent hallucination
4. format_instructions tell the LLM exactly what to produce
5. parser.parse() converts raw text to typed dict

Why Use Structured Output:
--------------------------
1. Reliability: Consistent, parseable responses
2. Type Safety: Pydantic validates the output
3. Integration: Easy to use in typed systems
4. Debugging: Clear schema makes issues obvious
5. Documentation: Schema serves as docs

Best Practices:
---------------
1. Use descriptive Field descriptions
2. Make fields optional when data may be absent
3. Tell LLM to use null for missing data
4. Use temperature=0 for more consistent output
5. Validate parsed output before using

Common Pitfalls:
----------------
1. Forgetting to call parser.parse() on the response
2. Not including format_instructions in the prompt
3. Required fields for data that may not exist
4. Not handling parse errors (LLM may not follow schema)

Error Handling Pattern:
-----------------------
try:
    parsed = parser.parse(llm_output)
except Exception as e:
    # Handle malformed JSON or schema mismatch
    logger.error(f"Failed to parse LLM output: {e}")
    parsed = None

Production Pattern:
-------------------
class StructuredExtractor:
    def __init__(self, llm: ChatOllama, model: type[BaseModel]):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=model)

    def extract(self, text: str, template: str) -> dict:
        prompt = ChatPromptTemplate.from_template(template)
        formatted = prompt.format_messages(
            text=text,
            format_instructions=self.parser.get_format_instructions()
        )
        response = self.llm.invoke(formatted)
        return self.parser.parse(response.content)

# Usage:
extractor = StructuredExtractor(llm, ReviewExtraction)
result = extractor.extract(customer_review, review_template)

Use Cases:
----------
- Data extraction from documents
- Form filling from natural language
- API response structuring
- Log parsing and analysis
- Resume/CV parsing
- Receipt/invoice extraction
- Entity extraction from text
"""
