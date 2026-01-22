# JSON Structured Output
# Extract structured data from text using Pydantic and JsonOutputParser

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

class ReviewExtraction(BaseModel):
    """Structured extraction from customer reviews."""
    gift: bool | None = Field(default=None, description="Was it purchased as a gift?")
    delivery_days: int | None = Field(default=None, description="Days for delivery")
    price_value: str | None = Field(default=None, description="Opinion on price vs value")

customer_review = """
I bought the Acme 3000 blender as a gift for my mom's birthday, \
and she absolutely loves it! I ordered it online and it arrived in just 3 days. \
It cost more than I expected, but the performance justifies the price.
"""

parser = JsonOutputParser(pydantic_object=ReviewExtraction)

review_template = """\
Analyze the following customer review and extract the required information. \
If information is not present in the review, use null for that field. \
{format_instructions}
text: ```{customer_review}```
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
formatted_prompt = prompt_template.format_messages(
    customer_review=customer_review,
    format_instructions=parser.get_format_instructions()
)

llm_output = llm.invoke(formatted_prompt).content
parsed_output = parser.parse(llm_output)

print(parsed_output)
# Output: {'gift': True, 'delivery_days': 3, 'price_value': 'expensive but worth it'}
