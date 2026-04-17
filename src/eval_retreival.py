import json
import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Setup
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "framework_docs"
RETRIEVAL_EVAL_PATH = "data/retrieval_eval.json"
RESULTS_PATH = "results/retrieval_eval_results.json"

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def embed_query(text: str):
    # Use the same logic from your embed_chunks.py script
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "task_type": "RETRIEVAL_QUERY", 
            "output_dimensionality": 768
        }
    )
    return response.embeddings[0].values

def get_retrieval(query, entity, top_k=3):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    # 1. Manually embed the query using Gemini
    query_embedding = embed_query(query)
    
    # 2. Query using the embedding, NOT the text
    results = collection.query(
        query_embeddings=[query_embedding], # Change this from query_texts
        n_results=top_k,
        where={"framework": entity} 
    )
    return results

def run_evaluation():
    with open(RETRIEVAL_EVAL_PATH, "r") as f:
        tests = json.load(f)

    results_report = []
    
    for test in tests:
        query = test["query"]
        entity = test["entity"]
        expected_topic = test["expected_topic"]
        
        print(f"Evaluating: [{entity}] {query}")
        
        # Perform retrieval
        raw_results = get_retrieval(query, entity)
        
        # Analyze results
        metadatas = raw_results['metadatas'][0]
        top1_topic = metadatas[0]['topic'] if metadatas else "None"
        
        top3_topics = [m['topic'] for m in metadatas]
        
        eval_entry = {
            "query": query,
            "entity": entity,
            "expected_topic": expected_topic,
            "top1_topic": top1_topic,
            "top1_correct": top1_topic == expected_topic,
            "top3_success": expected_topic in top3_topics,
            "top1_text_preview": raw_results['documents'][0][0][:100] + "..." if raw_results['documents'][0] else ""
        }
        results_report.append(eval_entry)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_report, f, indent=4)
    
    print(f"\nEvaluation Complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()