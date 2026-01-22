# Usage Metadata & Token Tracking
# Monitor input/output tokens for cost tracking

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

messages = [
    SystemMessage(content="You are an assistant who responds in the style of Shakespeare."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

response = llm.invoke(messages)

print(f"Input tokens: {response.usage_metadata.get('input_tokens', 'N/A')}")
print(f"Output tokens: {response.usage_metadata.get('output_tokens', 'N/A')}")
print(f"Content: {response.content}")
