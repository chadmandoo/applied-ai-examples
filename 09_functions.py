# =============================================================================
# Functions - LLM Function Calling
# =============================================================================
# Function calling lets LLMs decide which function to call based on user intent.
# The LLM parses the request, selects the appropriate function, and provides args.
#
# Key concepts:
#   - Define functions as tools
#   - LLM parses user intent to JSON
#   - Map function names to implementations
#   - Execute the selected function
#
# Corresponding endpoint: GET /api/functions/intent-parsing
# =============================================================================

from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)


# Define available functions
@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city (mock data)."""
    weather_data = {
        "london": {"temp": 58, "condition": "cloudy"},
        "tokyo": {"temp": 68, "condition": "sunny"},
    }
    return weather_data.get(city.lower(), {"temp": 65, "condition": "unknown"})


@tool
def calculate_date(days_from_now: int) -> str:
    """Calculate a date by adding days to today."""
    target = datetime.now() + timedelta(days=days_from_now)
    return target.strftime("%A, %B %d, %Y")


# Define schema for parsed function call
class FunctionCall(BaseModel):
    function_name: str = Field(description="Name of the function to call")
    arguments: dict = Field(description="Arguments for the function")


# Create parser
parser = JsonOutputParser(pydantic_object=FunctionCall)

# Prompt that instructs LLM to parse user intent
prompt = ChatPromptTemplate.from_template(
    """You are a function router. Determine which function to call.

Available functions:
1. get_weather(city: str) - Get weather for a city
2. calculate_date(days_from_now: int) - Calculate a future date

{format_instructions}

User request: {user_input}"""
)

chain = prompt | llm | parser

# Parse user request
user_request = "What's the weather like in London?"
result = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "user_input": user_request
})

print(f"User request: {user_request}")
print(f"Parsed intent: {result}")

# Execute the function
function_map = {"get_weather": get_weather, "calculate_date": calculate_date}
if result["function_name"] in function_map:
    output = function_map[result["function_name"]].invoke(result["arguments"])
    print(f"Function output: {output}")

# Example output:
# User request: What's the weather like in London?
# Parsed intent: {'function_name': 'get_weather', 'arguments': {'city': 'London'}}
# Function output: {'temp': 58, 'condition': 'cloudy'}
