# =============================================================================
# Document Retrieval - RAG Pattern (Stubbed)
# =============================================================================
# RAG (Retrieval Augmented Generation) retrieves relevant documents and uses
# them as context for LLM responses. This example uses mock data - connect
# to Milvus or Pinecone for production semantic search.
#
# Key concepts:
#   - Document store: Collection of text chunks
#   - Similarity search: Find relevant documents
#   - Context injection: Add retrieved docs to prompt
#   - Vector DB: Milvus, Pinecone, etc. (not shown)
#
# Corresponding endpoint: GET /api/retrieval/rag
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

# Mock document store (replace with vector DB in production)
DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "metadata": {"source": "programming_guide.txt"}
    },
    {
        "id": "doc2",
        "content": "FastAPI is a modern web framework for building APIs with Python, known for high performance.",
        "metadata": {"source": "web_frameworks.txt"}
    },
    {
        "id": "doc3",
        "content": "Vector databases store data as high-dimensional vectors for semantic search.",
        "metadata": {"source": "databases.txt"}
    },
]


def mock_similarity_search(query: str, top_k: int = 2) -> list[dict]:
    """
    Mock similarity search - replace with vector DB query.

    In production:
    1. Embed query using embedding model
    2. Search vector DB for similar vectors
    3. Return top_k documents
    """
    query_lower = query.lower()
    scored = []
    for doc in DOCUMENTS:
        # Simple keyword scoring (replace with cosine similarity)
        score = sum(1 for word in query_lower.split() if word in doc["content"].lower())
        if score > 0:
            scored.append({**doc, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# RAG Pipeline
query = "What is Python?"

# Step 1: Retrieve relevant documents
retrieved = mock_similarity_search(query, top_k=2)
print(f"Retrieved {len(retrieved)} documents")

# Step 2: Format context from documents
context = "\n\n".join([
    f"[{doc['metadata']['source']}]: {doc['content']}"
    for doc in retrieved
])

# Step 3: Generate answer with context
prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
)

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": query})

print(f"Query: {query}")
print(f"Answer: {answer}")

# Note: For production, connect to Milvus or Pinecone for semantic search

# Example output:
# Retrieved 2 documents
# Query: What is Python?
# Answer: Based on the context, Python is a high-level programming language...
