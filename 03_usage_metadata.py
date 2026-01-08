"""
Example 3: Usage Metadata & Token Tracking

This example demonstrates how to access usage metadata from LLM responses.
Tracking token usage is crucial for:
- Cost estimation and budgeting
- Performance monitoring
- Rate limiting
- Optimization decisions

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

# Create messages with system persona
messages = [
    SystemMessage(content="You are an assistant who responds in the style of Shakespeare."),
    HumanMessage(content="Write me a very short poem about a happy squirrel"),
]

# Invoke the LLM - this time we'll examine the full response object
response = llm.invoke(messages)

# Access usage metadata
input_tokens = response.usage_metadata.get('input_tokens', 'N/A')
output_tokens = response.usage_metadata.get('output_tokens', 'N/A')
content = response.content

# Display all information
print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Total tokens: {input_tokens + output_tokens if input_tokens != 'N/A' else 'N/A'}")
print(f"\nContent:\n{content}")

# Example output:
# Input tokens: 45
# Output tokens: 32
# Total tokens: 77
#
# Content:
# Hark! A creature most merry and spry,
# With bushy tail reaching up to the sky!

"""
Key Takeaways:
--------------
1. response.usage_metadata contains token counts
2. input_tokens = prompt tokens (what you sent)
3. output_tokens = completion tokens (what was generated)
4. Always use .get() with defaults for safety

Usage Metadata Fields:
----------------------
- input_tokens: Tokens in your prompt (system + user messages)
- output_tokens: Tokens in the LLM's response
- total_tokens: Sum of input + output (sometimes provided)

Why Track Tokens?
-----------------
1. Cost Management:
   - Most LLM APIs charge per token
   - OpenAI: $0.002 per 1K tokens (example)
   - Track usage to prevent budget overruns

2. Performance Optimization:
   - Shorter prompts = faster responses
   - Identify inefficient prompts
   - Optimize token usage

3. Rate Limiting:
   - Many APIs have token-per-minute limits
   - Track to avoid rate limit errors
   - Implement client-side throttling

4. User Experience:
   - Longer responses = slower generation
   - Balance detail vs. speed
   - Estimate response times

Cost Estimation Example:
------------------------
# Assume pricing: $0.002 per 1K tokens
input_tokens = 45
output_tokens = 32
total_tokens = 77

cost_per_1k = 0.002  # dollars
cost = (total_tokens / 1000) * cost_per_1k
print(f"Estimated cost: ${cost:.6f}")
# Output: Estimated cost: $0.000154

Production Pattern:
-------------------
def track_llm_usage(response):
    usage = {
        "input_tokens": response.usage_metadata.get('input_tokens', 0),
        "output_tokens": response.usage_metadata.get('output_tokens', 0),
        "timestamp": datetime.now(timezone.utc),
        "model": "llama3.2"
    }

    # Log to database or monitoring service
    log_usage(usage)

    return usage

Use Cases:
----------
- Production cost tracking
- User quota management
- Performance monitoring dashboards
- Billing and invoicing
- Optimization and A/B testing
"""
