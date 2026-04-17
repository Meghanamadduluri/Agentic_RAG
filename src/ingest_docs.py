import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import SOURCE_PAGES
from utils import (
    ensure_directories,
    fetch_html,
    extract_main_text,
    clean_text,
    save_json,
    save_text,
    slugify,
)


def chunk_documents(documents, chunk_size=900, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        paragraph_blocks = []
        current_block = ""

        for para in paragraphs:
            if len(current_block) + len(para) + 2 <= chunk_size:
                current_block += ("\n\n" + para) if current_block else para
            else:
                if current_block:
                    paragraph_blocks.append(current_block)
                current_block = para

        if current_block:
            paragraph_blocks.append(current_block)

        final_chunks = []
        for block in paragraph_blocks:
            if len(block) <= chunk_size:
                final_chunks.append(block)
            else:
                final_chunks.extend(splitter.split_text(block))

        deduped_chunks = []
        seen_normalized = set()

        for chunk in final_chunks:
            chunk = chunk.strip()
            if len(chunk) < 180:
                continue

            normalized = re.sub(r"\s+", " ", chunk.lower()).strip()

            # light exact dedupe
            if normalized in seen_normalized:
                continue

            seen_normalized.add(normalized)
            deduped_chunks.append(chunk)

        for idx, chunk in enumerate(deduped_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{idx+1}",
                    "framework": doc["framework"],
                    "topic": doc["topic"],
                    "title": doc["title"],
                    "source_url": doc["source_url"],
                    "text": chunk,
                }
            )

    return all_chunks


def main():
    ensure_directories()

    processed_docs = []

    for page in SOURCE_PAGES:
        framework = page["framework"]
        topic = page["topic"]
        url = page["url"]

        print(f"\nFetching: {framework} | {topic}")
        print(f"URL: {url}")

        try:
            html = fetch_html(url)
            raw_filename = f"{slugify(framework)}_{slugify(topic)}.html"
            raw_path = os.path.join("data/raw", raw_filename)

            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(html)

            text = extract_main_text(html)
            text = clean_text(text)

            if len(text) < 300:
                print(f"Skipped: extracted text too short for {url}")
                continue

            doc_id = f"{slugify(framework)}_{slugify(topic)}"
            title = f"{framework} {topic.title()}"

            txt_filename = f"{doc_id}.txt"
            txt_path = os.path.join("data/processed", txt_filename)
            save_text(txt_path, text)

            processed_docs.append(
                {
                    "doc_id": doc_id,
                    "framework": framework,
                    "topic": topic,
                    "title": title,
                    "source_url": url,
                    "text": text,
                }
            )

            print(f"Saved cleaned text: {txt_path}")
            print(f"Extracted characters: {len(text)}")

        except Exception as e:
            print(f"Failed on {url}")
            print(f"Error: {e}")

    save_json("data/processed/documents.json", processed_docs)
    print(f"\nSaved {len(processed_docs)} processed documents.")

    chunks = chunk_documents(processed_docs)
    save_json("data/processed/chunks.json", chunks)

    print(f"Saved {len(chunks)} chunks to data/processed/chunks.json")


if __name__ == "__main__":
    main()