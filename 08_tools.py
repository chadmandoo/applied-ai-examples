# =============================================================================
# Tools - Giving LLMs External Capabilities
# =============================================================================
# Tools allow LLMs to interact with external systems: APIs, databases,
# calculators, etc. Define tools with the @tool decorator, then invoke them.
#
# Key concepts:
#   - @tool decorator: Create tools from functions
#   - Tool schema: Auto-generated from function signature
#   - invoke(): Execute the tool
#   - Pydantic args_schema: Complex input validation
#
# Corresponding endpoint: GET /api/tools/basic
# =============================================================================

import math
from datetime import datetime
from langchain_core.tools import tool

# Define a simple tool using the @tool decorator
# The docstring becomes the tool description (important for LLM understanding)
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * radius ** 2


@tool
def convert_temperature(celsius: float) -> dict:
    """Convert Celsius to Fahrenheit and Kelvin."""
    fahrenheit = (celsius * 9 / 5) + 32
    kelvin = celsius + 273.15
    return {
        "celsius": celsius,
        "fahrenheit": round(fahrenheit, 2),
        "kelvin": round(kelvin, 2)
    }


# Execute tools directly
current_time = get_current_time.invoke({})
print(f"Current time: {current_time}")

circle_area = calculate_circle_area.invoke({"radius": 5.0})
print(f"Circle area (r=5): {circle_area:.2f}")

temp = convert_temperature.invoke({"celsius": 25.0})
print(f"Temperature conversion: {temp}")

# View tool schema
print(f"\nTool name: {calculate_circle_area.name}")
print(f"Description: {calculate_circle_area.description}")

# Example output:
# Current time: 2024-01-15 14:30:45
# Circle area (r=5): 78.54
# Temperature conversion: {'celsius': 25.0, 'fahrenheit': 77.0, 'kelvin': 298.15}
