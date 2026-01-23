# =============================================================================
# Agents - LLMs That Decide and Act
# =============================================================================
# Agents are LLMs that can decide which tools to use and in what order.
# They follow the ReAct pattern: Reasoning + Acting in a loop until done.
#
# Key concepts:
#   - ReAct: Thought → Action → Observation → Repeat
#   - Agent decides which tool to use
#   - Tools are executed and results fed back
#   - Loop until final answer is reached
#
# Corresponding endpoint: GET /api/agents/react-simple
# =============================================================================

import math
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Initialize the Ollama client
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)


# Define agent tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like '2 + 2' or 'sqrt(16)'."""
    allowed = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs}
    return str(eval(expression, {"__builtins__": {}}, allowed))


@tool
def get_date_info() -> dict:
    """Get current date and time information."""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "day": now.strftime("%A")}


# Schema for agent's reasoning
class AgentThought(BaseModel):
    thought: str = Field(description="What the agent is thinking")
    action: str = Field(description="'use_tool' or 'final_answer'")
    tool_name: str | None = Field(default=None, description="Tool to use")
    tool_input: dict | None = Field(default=None, description="Tool arguments")
    final_answer: str | None = Field(default=None, description="Final answer")


parser = JsonOutputParser(pydantic_object=AgentThought)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant with access to tools.

Available tools:
1. calculator(expression: str) - Evaluate math like 'sqrt(16)'
2. get_date_info() - Get current date/time

Think step by step. If you need a tool, use it. If you can answer, provide final_answer.

{format_instructions}

Question: {question}"""
)

chain = prompt | llm | parser

# Agent reasoning
question = "What is the square root of 144?"
thought = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "question": question
})

print(f"Question: {question}")
print(f"Agent thought: {thought['thought']}")
print(f"Action: {thought['action']}")

# Execute tool if needed
if thought["action"] == "use_tool" and thought["tool_name"]:
    tools = {"calculator": calculator, "get_date_info": get_date_info}
    result = tools[thought["tool_name"]].invoke(thought["tool_input"] or {})
    print(f"Tool result: {result}")

# Example output:
# Question: What is the square root of 144?
# Agent thought: I need to calculate sqrt(144)
# Action: use_tool
# Tool result: 12.0
