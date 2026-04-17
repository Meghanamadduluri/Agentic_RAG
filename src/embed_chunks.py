import json
import os
import time
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


def load_chunks(path="data/processed/chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_text(text: str):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 768
        }
    )
    return response.embeddings[0].values


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    # ... previous code ...

    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}: {chunk['chunk_id']}")

        try:
            embedding = embed_text(chunk["text"])
        except Exception as e:
            print(f"Error embedding chunk {chunk['chunk_id']}: {e}")
            print("Sleeping for 60 seconds before retry...")
            time.sleep(60)
            embedding = embed_text(chunk["text"])

        # THESE MUST BE INDENTED to be part of the for-loop
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        embeddings.append(embedding)
        metadatas.append(
            {
                "framework": chunk["framework"],
                "topic": chunk["topic"],
                "title": chunk["title"],
                "source_url": chunk["source_url"],
            }
        )

        # Throttling inside the loop
        time.sleep(1.5) 

    # Now, outside the loop, we add EVERYTHING at once
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(ids)} chunks in ChromaDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()