# =============================================================================
# Memory - Conversation State with SQLite
# =============================================================================
# Memory allows LLMs to maintain context across interactions. This example
# uses SQLite to persist conversation history, enabling stateful conversations.
#
# Key concepts:
#   - Store messages in SQLite database
#   - Retrieve history for context
#   - Format history for prompt inclusion
#   - Window memory: Only use last K messages
#
# Corresponding endpoint: GET /api/memory/continue
# =============================================================================

import sqlite3
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

# SQLite database path
DB_PATH = Path("./conversations.db")


def init_db():
    """Initialize database with sample conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)

    # Add sample history if empty
    cursor.execute("SELECT COUNT(*) FROM messages")
    if cursor.fetchone()[0] == 0:
        messages = [
            ("human", "What is Python?"),
            ("ai", "Python is a high-level programming language known for its simple syntax."),
            ("human", "What are its main uses?"),
            ("ai", "Python is used for web development, data science, and automation."),
        ]
        cursor.executemany("INSERT INTO messages (role, content) VALUES (?, ?)", messages)

    conn.commit()
    conn.close()


def get_history() -> list[dict]:
    """Retrieve conversation history from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY id")
    messages = [{"role": role, "content": content} for role, content in cursor.fetchall()]
    conn.close()
    return messages


def format_history(history: list[dict]) -> str:
    """Format history as string for prompt."""
    formatted = []
    for msg in history:
        prefix = "Human" if msg["role"] == "human" else "Assistant"
        formatted.append(f"{prefix}: {msg['content']}")
    return "\n".join(formatted)


# Initialize database
init_db()

# Get conversation history
history = get_history()
history_text = format_history(history)

# Create prompt with history context
prompt = ChatPromptTemplate.from_template(
    """Previous conversation:
{history}

Based on the conversation above, answer:
Human: {question}
Assistant:"""
)

chain = prompt | llm | StrOutputParser()

# Continue the conversation using history
result = chain.invoke({
    "history": history_text,
    "question": "Can you give me an example of Python automation?"
})

print(f"History used: {len(history)} messages")
print(f"Response: {result}")

# Example output: Based on our discussion about Python's uses...
