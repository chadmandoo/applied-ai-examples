# =============================================================================
# Few-Shot Prompting
# =============================================================================
# Provide examples of desired input/output pairs so the LLM learns the pattern.
# More effective than just giving instructions - showing beats telling.
#
# Key concepts:
#   - examples: List of input/output dictionaries
#   - example_prompt: Template for formatting each example
#   - FewShotPromptTemplate: Combines examples with the actual query
#   - prefix/suffix: Text before/after the examples
#
# Corresponding endpoint: GET /api/prompt-templates/few-shot
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Define examples showing input → output pattern
# The LLM learns to replicate this pattern
examples = [
    {"input": "The weather is beautiful today.", "output": "Positive"},
    {"input": "I'm feeling terrible and everything is going wrong.", "output": "Negative"},
    {"input": "The movie was okay, nothing special.", "output": "Neutral"},
]

# Template for formatting each example
example_template = """
Input: {input}
Sentiment: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

# Combine everything into a few-shot prompt
# - prefix: Instructions before examples
# - examples: The example data
# - example_prompt: How to format each example
# - suffix: The actual query template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of the following texts as Positive, Negative, or Neutral.\n\nExamples:",
    suffix="\nInput: {input}\nSentiment:",
    input_variables=["input"]
)

# Format the prompt with the new input
# The LLM sees the examples first, then classifies the new input
formatted_prompt = few_shot_prompt.format(input="The service was disappointing and slow.")

# Invoke the LLM
response = llm.invoke(formatted_prompt)

print(f"Input: 'The service was disappointing and slow.'")
print(f"Predicted Sentiment: {response.content}")

# Example output: Negative
