import json
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

PLANNER_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def plan_task(query):
    """
    Decomposes a raw query into a structured execution plan.
    """
    prompt = f"""
    SYSTEM: You are a technical task planner for an Agentic RAG system.
    Your job is to identify the entities (LangChain or LlamaIndex) and the specific technical aspects to research.

    TASK_TYPES: 
    - "comparison": Comparing two or more entities.
    - "decision": Recommending one entity over another.
    - "lookup": Detailed deep-dive into one entity.

    QUERY: {query}

    OUTPUT FORMAT (JSON ONLY):
    {{
      "task_type": "comparison" | "decision" | "lookup",
      "entities": ["LangChain", "LlamaIndex"],
      "aspects": ["list", "of", "technical", "topics", "to", "retrieve"]
    }}

    EXAMPLE:
    Query: "Compare memory management in LangChain vs LlamaIndex"
    Output: {{
      "task_type": "comparison",
      "entities": ["LangChain", "LlamaIndex"],
      "aspects": ["short-term memory", "persistence", "chat history"]
    }}
    """

    response = client.models.generate_content(
        model=PLANNER_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "task_type": "comparison",
            "entities": ["LangChain", "LlamaIndex"],
            "aspects": ["general architecture"]
        }

if __name__ == "__main__":
    # Test the brain
    test_query = "How do LangChain and LlamaIndex handle tool definition for agents?"
    print(json.dumps(plan_task(test_query), indent=2))