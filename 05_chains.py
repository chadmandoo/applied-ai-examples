# =============================================================================
# Chains - Basic Chain Composition
# =============================================================================
# Chains connect prompts, LLMs, and output parsers into a pipeline.
# Using LCEL (LangChain Expression Language), you compose chains with the
# pipe operator (|) for clean, readable code.
#
# Key concepts:
#   - prompt | llm | parser: The basic chain pattern
#   - invoke(): Execute the chain with input data
#   - Sequential chains: Output of one feeds into the next
#
# Corresponding endpoint: GET /api/chains/basic
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create a prompt template with a variable placeholder
prompt = ChatPromptTemplate.from_template(
    "Tell me a short fact about {topic}"
)

# Build the chain using LCEL pipe operator
# prompt → LLM → output parser
chain = prompt | llm | StrOutputParser()

# Execute the chain with input data
result = chain.invoke({"topic": "Python programming"})

print(result)

# Example output: Python was created by Guido van Rossum and first released in 1991...
