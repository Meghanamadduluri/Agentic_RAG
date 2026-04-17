import json
import os
import time
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Minimal Constants
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "framework_docs"

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_baseline_retrieval(entities, query):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    # 1. Embed query
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768}
    )
    query_embedding = response.embeddings[0].values

    context_text = ""
    sources = []

    # 2. Simple retrieval per entity
    for entity in entities:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"framework": entity}
        )
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_text += f"\n[SOURCE: {meta['framework']}]\nCONTENT: {doc[:800]}\n"
            sources.append(meta['title'])
            
    return context_text, list(set(sources))

def generate_comparison(query, entities, context):
    prompt = f"""
    Compare {entities} based ONLY on this evidence:
    {context}

    Return a JSON object:
    {{
      "query": "{query}",
      "entities": {json.dumps(entities)},
      "comparison": {{
        "retrieval": "compare retrieval logic",
        "agents": "compare agent support",
        "developer_experience": "compare ease of use"
      }},
      "recommendation": "which one to use and why",
      "sources_used": {json.dumps(entities)}
    }}
    """
    
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

if __name__ == "__main__":
    # Add a simple choice at the start
    print("1. Run Batch Evaluation")
    print("2. Ask a Single Question")
    choice = input("Select mode (1 or 2): ")

    if choice == "1":
        # --- THIS IS YOUR BENCHMARK TOOL (STAYS UNCHANGED) ---
        with open("data/test_eval.json", "r") as f:
            tasks = json.load(f)
        
        results = []
        for task in tasks[:6]:
            print(f"Processing Batch Task: {task['query']}")
            try:
                context, sources = get_baseline_retrieval(task['entities'], task['query'])
                output = generate_comparison(task['query'], task['entities'], context)
                output["actual_sources"] = sources
                results.append(output)
                time.sleep(12) 
            except Exception as e:
                print(f"Failed: {e}")

        with open("results/baseline_task_outputs.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Done. Batch results saved.")

    else:
        # --- THIS IS YOUR NEW INTERACTIVE MODE ---
        user_query = input("Enter your technical question: ")
        # In baseline, we manually define the entities for now
        entities = ["LangChain", "LlamaIndex"] 
        
        print(f"Running Baseline Research for: {user_query}")
        context, sources = get_baseline_retrieval(entities, user_query)
        result = generate_comparison(user_query, entities, context)
        
        print("\n--- BASELINE RESULT ---")
        print(json.dumps(result, indent=2))