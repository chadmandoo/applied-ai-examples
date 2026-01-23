# =============================================================================
# Routing - Dynamic Chain Selection
# =============================================================================
# Routing directs inputs to different chains based on classification.
# Use RunnableBranch for conditional logic or classify-then-route patterns.
#
# Key concepts:
#   - Classify input to determine route
#   - RunnableBranch: Conditional chain selection
#   - Different prompts/chains for different categories
#   - Fallback routes for unmatched inputs
#
# Corresponding endpoint: GET /api/routing/classify
# =============================================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)


# Classification schema
class RouteClassification(BaseModel):
    category: str = Field(description="'technical', 'creative', or 'factual'")
    confidence: str = Field(description="'high', 'medium', or 'low'")


parser = JsonOutputParser(pydantic_object=RouteClassification)

# Classifier prompt
classify_prompt = ChatPromptTemplate.from_template(
    """Classify this question into one category:
- technical: Programming, software, computers
- creative: Writing, art, storytelling
- factual: History, geography, general knowledge

{format_instructions}

Question: {question}"""
)

# Specialized prompts for each category
prompts = {
    "technical": ChatPromptTemplate.from_template(
        "You are a technical expert. Provide a clear answer with examples:\n\n{question}"
    ),
    "creative": ChatPromptTemplate.from_template(
        "You are a creative writer. Be imaginative:\n\n{question}"
    ),
    "factual": ChatPromptTemplate.from_template(
        "You are knowledgeable. Provide accurate information:\n\n{question}"
    ),
}

# Classify the input
question = "How do I write a for loop in Python?"

classifier = classify_prompt | llm | parser
classification = classifier.invoke({
    "format_instructions": parser.get_format_instructions(),
    "question": question
})

print(f"Question: {question}")
print(f"Classification: {classification}")

# Route to appropriate prompt
category = classification["category"]
if category in prompts:
    response_chain = prompts[category] | llm | StrOutputParser()
    response = response_chain.invoke({"question": question})
    print(f"Routed to: {category}")
    print(f"Response: {response[:200]}...")

# Example output:
# Question: How do I write a for loop in Python?
# Classification: {'category': 'technical', 'confidence': 'high'}
# Routed to: technical
# Response: A for loop in Python iterates over a sequence...
