"""
Example 2: System Messages & Persona

This example shows how to use system messages to give the LLM a persona or
specific instructions. System messages are used to:
- Set the LLM's role or personality
- Provide context and constraints
- Define response style and format

⚠️ This is a non-compilable example for educational purposes.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Initialize the LLM client
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.5
)

# Create a message array with system instructions and user query
messages = [
    # System message sets the LLM's behavior and personality
    SystemMessage(content="You are an assistant who responds in the style of Dr Seuss."),

    # Human message is the actual user query
    HumanMessage(content="write me a very short poem about a happy squirrel"),
]

# Invoke with both system and user messages
response = llm.invoke(messages)
print(response.content)

# Example output (Dr. Seuss style):
# "Oh my! Oh me! A squirrel so bright,
#  With fluffy tail dancing in morning light!"

# ============================================================================
# Example 2b: Different Persona (Shakespeare)
# ============================================================================

messages_shakespeare = [
    SystemMessage(content="You are an assistant who responds in the style of Shakespeare."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

response_shakespeare = llm.invoke(messages_shakespeare)
print(response_shakespeare.content)

# Example output (Shakespeare style):
# "Hark! A creature most merry and spry,
#  With bushy tail reaching up to the sky!"

"""
Key Takeaways:
--------------
1. SystemMessage provides instructions to the LLM
2. Message order matters: System messages typically come first
3. You can combine multiple messages in one request
4. The persona/style persists throughout the conversation

Message Types:
--------------
- SystemMessage: Instructions for the LLM (persona, rules, context)
- HumanMessage: User input/queries
- AIMessage: Previous AI responses (for conversation history)

Best Practices:
---------------
1. Keep system messages clear and specific
2. Put system messages before human messages
3. Don't contradict yourself in system messages
4. Test different phrasings for best results

Use Cases:
----------
- Customer service bots with specific tone
- Code reviewers with strict guidelines
- Creative writing in specific styles
- Domain-specific experts (medical, legal, technical)
- Consistent brand voice
"""
