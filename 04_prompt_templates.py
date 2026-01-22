# Prompt Templates
# Create reusable, parameterized prompts

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.5
)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```

Return only the translated text.
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

formatted_prompt = prompt_template.format_messages(
    style="slang",
    text="Hello how art thou?"
)

response = llm.invoke(formatted_prompt)
print(response.content)
