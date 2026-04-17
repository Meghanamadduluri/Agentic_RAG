import os
from dotenv import load_dotenv
import chromadb
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "framework_docs"
EMBED_MODEL = "gemini-embedding-001"


def embed_query(query: str):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": 768
        }
    )
    return response.embeddings[0].values

def detect_framework_filter(query: str):
    q = query.lower()

    if "langchain" in q:
        return {"framework": "LangChain"}
    if "llamaindex" in q or "llama index" in q:
        return {"framework": "LlamaIndex"}

    return None

def retrieve(query: str, top_k: int = 5):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    query_embedding = embed_query(query)
    framework_filter = detect_framework_filter(query)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if framework_filter:
        query_args["where"] = framework_filter

    results = collection.query(**query_args)
    return results


def print_results(query: str, results):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        print(f"\nResult {i}")
        print(f"Framework : {meta['framework']}")
        print(f"Topic     : {meta['topic']}")
        print(f"Title     : {meta['title']}")
        print(f"Source    : {meta['source_url']}")
        print(f"Distance  : {distance}")
        print(f"Content   : {doc[:700]}")
        print("-" * 80)


def main():
    sample_query = "How does Langchain handle indexing?"
    results = retrieve(sample_query, top_k=3)
    print_results(sample_query, results)


if __name__ == "__main__":
    main()