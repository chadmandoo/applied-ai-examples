# =============================================================================
# LangGraph - Stateful Workflows with SQLite
# =============================================================================
# LangGraph enables complex, stateful workflows with conditional branching.
# This example uses SQLite to persist workflow state across steps.
#
# Key concepts:
#   - Workflow state: Track progress through nodes
#   - Conditional branching: Route based on analysis
#   - State persistence: SQLite for durability
#   - Multi-step processing: Chain of dependent nodes
#
# Corresponding endpoint: GET /api/langgraph/basic-flow
# =============================================================================

import sqlite3
import json
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

# SQLite for workflow state persistence
DB_PATH = Path("./workflows.db")


def init_db():
    """Initialize workflow database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflow_steps (
            id INTEGER PRIMARY KEY,
            workflow_id TEXT,
            step_name TEXT,
            step_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_step(workflow_id: str, step_name: str, data: dict):
    """Save workflow step to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO workflow_steps (workflow_id, step_name, step_data) VALUES (?, ?, ?)",
        (workflow_id, step_name, json.dumps(data))
    )
    conn.commit()
    conn.close()


# Initialize database
init_db()

# Workflow ID for this run
import uuid
workflow_id = str(uuid.uuid4())[:8]

# Input to process
input_text = "I need help with my recent order. It hasn't arrived yet."

# === NODE 1: Classify ===
classify_prompt = ChatPromptTemplate.from_template(
    """Classify this customer message into one category: 'support', 'sales', or 'general'.
Return only the category word.

Message: {text}"""
)

classify_chain = classify_prompt | llm | StrOutputParser()
classification = classify_chain.invoke({"text": input_text}).strip().lower()

save_step(workflow_id, "classify", {"input": input_text, "output": classification})
print(f"Step 1 - Classification: {classification}")

# === NODE 2: Route and Respond ===
# Conditional branching based on classification
response_prompts = {
    "support": "You are a support agent. Address this concern helpfully: {text}",
    "sales": "You are a sales rep. Respond to this inquiry: {text}",
    "general": "You are a helpful assistant. Respond to: {text}"
}

template = response_prompts.get(classification, response_prompts["general"])
response_prompt = ChatPromptTemplate.from_template(template)
response_chain = response_prompt | llm | StrOutputParser()
response = response_chain.invoke({"text": input_text})

save_step(workflow_id, "respond", {"classification": classification, "response": response})
print(f"Step 2 - Response generated for '{classification}' route")

# === Final State ===
print(f"\nWorkflow {workflow_id} completed:")
print(f"  Input: {input_text[:50]}...")
print(f"  Route: {classification}")
print(f"  Response: {response[:100]}...")

# Example output:
# Step 1 - Classification: support
# Step 2 - Response generated for 'support' route
# Workflow abc12345 completed:
#   Input: I need help with my recent order...
#   Route: support
#   Response: I apologize for the inconvenience...
