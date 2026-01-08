"""
Example 4: Prompt Templates

This example demonstrates how to use ChatPromptTemplate to create reusable,
parameterized prompts. Prompt templates are essential for:
- Consistent prompt formatting
- Reusable prompt patterns
- Separating prompt logic from application code
- Easy prompt versioning and testing

⚠️ This is a non-compilable example for educational purposes.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Initialize the LLM client
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.5
)

# Define a reusable prompt template with variables
# Variables are enclosed in {curly_braces}
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```

Return only the translated text.
"""

# Create a ChatPromptTemplate from the template string
prompt_template = ChatPromptTemplate.from_template(template_string)

# Format the prompt with specific values
formatted_prompt = prompt_template.format_messages(
    style="slang",
    text="Hello how art thou?"
)

# Invoke the LLM with the formatted prompt
response = llm.invoke(formatted_prompt)
print(response.content)

# Example output:
# "Yo, what's good?"

# ============================================================================
# Example 4b: Different Parameters
# ============================================================================

# Reuse the same template with different parameters
formatted_prompt_formal = prompt_template.format_messages(
    style="formal business English",
    text="Hey dude, what's up?"
)

response_formal = llm.invoke(formatted_prompt_formal)
print(response_formal.content)

# Example output:
# "Good day, how may I assist you?"

# ============================================================================
# Example 4c: Complex Template with Multiple Variables
# ============================================================================

translation_template = """You are a professional translator specializing in {domain}.

Translate the following {source_lang} text into {target_lang}.
Maintain the {tone} tone.

Text to translate: ```{text}```

Provide only the translation, nothing else.
"""

prompt = ChatPromptTemplate.from_template(translation_template)

formatted = prompt.format_messages(
    domain="medical terminology",
    source_lang="English",
    target_lang="Spanish",
    tone="professional",
    text="The patient is experiencing acute abdominal pain."
)

response = llm.invoke(formatted)
print(response.content)

# Example output:
# "El paciente está experimentando dolor abdominal agudo."

"""
Key Takeaways:
--------------
1. ChatPromptTemplate creates reusable prompt patterns
2. Variables use {variable_name} syntax
3. format_messages() fills in the variables
4. Templates separate prompt structure from data

Template Benefits:
------------------
1. Reusability: Define once, use many times
2. Consistency: Same structure for all requests
3. Testability: Easy to test with different inputs
4. Maintainability: Update template in one place
5. Version Control: Track prompt changes over time

Advanced Template Features:
---------------------------
1. Multi-line templates (use triple quotes)
2. Multiple variables in one template
3. Optional variables with defaults
4. Template composition (combine templates)
5. System/Human message separation

Best Practices:
---------------
1. Use descriptive variable names ({user_query} not {q})
2. Include clear instructions in the template
3. Use delimiters (```) for user content
4. Specify output format in the template
5. Test templates with edge cases

Common Template Patterns:
-------------------------

# Translation Template
"Translate {text} from {source} to {target}"

# Summarization Template
"Summarize the following text in {num_sentences} sentences: {text}"

# Classification Template
"Classify this text as {categories}: {text}"

# Q&A Template
"Given this context: {context}\nAnswer the question: {question}"

# Code Review Template
"Review this {language} code for {aspects}: {code}"

Production Pattern:
-------------------
class PromptTemplates:
    TRANSLATION = ChatPromptTemplate.from_template(
        "Translate {text} into {style}"
    )

    SUMMARIZATION = ChatPromptTemplate.from_template(
        "Summarize in {length}: {text}"
    )

    @staticmethod
    def get_template(name: str) -> ChatPromptTemplate:
        return getattr(PromptTemplates, name.upper())

# Usage:
template = PromptTemplates.get_template("translation")
prompt = template.format_messages(text="Hello", style="formal")

Use Cases:
----------
- Translation services
- Content transformation
- Text classification
- Code generation
- Data extraction
- Multi-step workflows
- A/B testing different prompts
"""
