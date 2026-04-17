import json
import os
import time
from planner import plan_task
from baseline_compare import get_baseline_retrieval, generate_comparison, client

# Constants
GEN_MODEL = "gemini-2.5-flash"

def run_agentic_task(user_query):
    # 1. PLAN: Decompose the query
    print(f"--- [PLANNING PHASE] ---")
    plan = plan_task(user_query)
    print(f"Plan created: {plan['entities']} | Aspects: {plan['aspects']}")

    collected_context = ""
    all_sources = []
    
    # 2. DECOMPOSED RETRIEVAL: Loop through entities and aspects
    print(f"\n--- [RESEARCH PHASE] ---")
    for entity in plan["entities"]:
        for aspect in plan["aspects"]:
            search_query = f"{entity} {aspect} implementation details"
            print(f"Researching: {search_query}")
            
            # Retrieve specifically for this entity+aspect combo
            # We use top_k=2 because we are making many targeted calls
            context, sources = get_baseline_retrieval([entity], search_query)
            
            collected_context += f"\n### RESEARCH BLOCK: {entity} - {aspect}\n{context}\n"
            all_sources.extend(sources)
            
            # Quota Protection
            time.sleep(2) 

    # 3. SYNTHESIS: Final comparison based on organized research
    print(f"\n--- [SYNTHESIS PHASE] ---")
    final_result = generate_comparison(
        query=user_query,
        entities=plan["entities"],
        context=collected_context,
        #expected_aspects=plan["aspects"] # Pass dynamic aspects
    )
    
    final_result["actual_sources"] = list(set(all_sources))
    return final_result

if __name__ == "__main__":
    print("Agentic Technical Decision Support System Active.")
    query = input("Enter your technical question: ")
    
    if query:
        result = run_agentic_task(query)
        print("\n--- AGENTIC RESULT ---")
        print(json.dumps(result, indent=2))