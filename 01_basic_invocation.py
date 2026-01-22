# Basic LLM Invocation
# Send a simple message and get a response

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

messages = [HumanMessage(content="Say hello world in one sentence.")]
response = llm.invoke(messages)

print(response.content)
