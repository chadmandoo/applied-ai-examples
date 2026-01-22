# =============================================================================
# Multi-Variable Prompt Templates
# =============================================================================
# Handle multiple variables in a single prompt for complex, parameterized
# content generation. Great for form-based generation and personalized content.
#
# Key concepts:
#   - Multiple {variable} placeholders in one template
#   - All variables filled via format_messages()
#   - Enables dynamic, reusable content generation
#
# Corresponding endpoint: GET /api/prompt-templates/multi-variable
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Define a template with multiple placeholders
# Each {variable} will be replaced with actual values
template_string = """Write a {style} {content_type} about {subject} for {audience}.

The {content_type} should be approximately {length} words long and have a {tone} tone.

Make it engaging and appropriate for the target audience."""

# Create the prompt template
prompt_template = ChatPromptTemplate.from_template(template_string)

# Fill in all the variables at once
# This creates a formatted message list ready for the LLM
formatted_prompt = prompt_template.format_messages(
    style="professional",
    content_type="blog post",
    subject="artificial intelligence in healthcare",
    audience="medical professionals",
    length="100",
    tone="informative"
)

# Send to LLM and get response
response = llm.invoke(formatted_prompt)

print(response.content)

# Example output: A professional blog post about AI in healthcare...
