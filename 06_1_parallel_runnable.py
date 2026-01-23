# =============================================================================
# Parallel Runnable - Concurrent Execution
# =============================================================================
# RunnableParallel executes multiple chains concurrently, collecting results
# into a dictionary. Great for when you need multiple independent analyses.
#
# Key concepts:
#   - RunnableParallel: Run multiple chains at once
#   - Results returned as dict with named keys
#   - Each chain runs independently
#
# Corresponding endpoint: GET /api/runnable/parallel
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Define two different prompts for parallel execution
joke_prompt = ChatPromptTemplate.from_template(
    "Tell a very short joke about {topic}"
)
fact_prompt = ChatPromptTemplate.from_template(
    "Tell me one interesting fact about {topic}"
)

# Create individual chains
joke_chain = joke_prompt | llm | StrOutputParser()
fact_chain = fact_prompt | llm | StrOutputParser()

# Combine into parallel execution
# Both chains run at the same time with the same input
parallel_chain = RunnableParallel(
    joke=joke_chain,
    fact=fact_chain
)

# Execute - returns dict with 'joke' and 'fact' keys
result = parallel_chain.invoke({"topic": "cats"})

print(f"Joke: {result['joke']}")
print(f"Fact: {result['fact']}")

# Example output:
# Joke: Why don't cats play poker? Too many cheetahs!
# Fact: Cats spend 70% of their lives sleeping.
