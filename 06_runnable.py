# =============================================================================
# Runnable Interface - LCEL Basics
# =============================================================================
# The Runnable interface is the foundation of LCEL (LangChain Expression Language).
# It provides a standard interface for all components: invoke, batch, stream.
#
# Key concepts:
#   - invoke(): Single input execution
#   - batch(): Process multiple inputs
#   - RunnablePassthrough: Keep original input while transforming
#   - RunnableParallel: Run multiple chains concurrently
#
# Corresponding endpoint: GET /api/runnable/passthrough
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# Create a prompt that needs both original input and added context
prompt = ChatPromptTemplate.from_template(
    """Original question: {question}
Context: {context}

Answer the question based on the context."""
)

# RunnablePassthrough keeps the original input ('question')
# while we add new data ('context') via a lambda
chain = (
    {
        "question": RunnablePassthrough(),
        "context": lambda x: "Python is a programming language created by Guido van Rossum."
    }
    | prompt
    | llm
    | StrOutputParser()
)

# The original question is preserved and passed through
result = chain.invoke("Who created Python?")

print(result)

# Example output: Based on the context, Python was created by Guido van Rossum.
