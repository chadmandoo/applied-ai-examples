# =============================================================================
# Sequential Chains
# =============================================================================
# Sequential chains pass the output of one chain as input to the next.
# This enables multi-step processing where each step builds on the previous.
#
# Key concepts:
#   - Chain 1 output → Chain 2 input
#   - Each chain can have different prompts/purposes
#   - Great for decomposing complex tasks
#
# Corresponding endpoint: GET /api/chains/sequential
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

# Chain 1: Generate a topic
prompt1 = ChatPromptTemplate.from_template(
    "Generate a single interesting topic about {subject}. Return only the topic, nothing else."
)
chain1 = prompt1 | llm | StrOutputParser()

# Chain 2: Write about that topic
prompt2 = ChatPromptTemplate.from_template(
    "Write a brief 2-sentence explanation about: {topic}"
)
chain2 = prompt2 | llm | StrOutputParser()

# Execute sequentially - output of chain1 feeds into chain2
topic = chain1.invoke({"subject": "space exploration"})
print(f"Generated topic: {topic}")

explanation = chain2.invoke({"topic": topic})
print(f"Explanation: {explanation}")

# Example output:
# Generated topic: The Voyager Golden Record
# Explanation: The Voyager Golden Record is a phonograph record...
