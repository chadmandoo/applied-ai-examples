# System Messages & Persona
# Add personality/style to LLM responses using SystemMessage

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

messages = [
    SystemMessage(content="You are an assistant who responds in the style of Dr Seuss."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

response = llm.invoke(messages)
print(response.content)
