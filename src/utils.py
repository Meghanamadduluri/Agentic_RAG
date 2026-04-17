import json
import os
import re
from typing import List, Dict

import requests
import trafilatura
from bs4 import BeautifulSoup


def ensure_directories():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AgenticRAGProject/1.0)"
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.text


def extract_main_text(html: str) -> str:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted and len(extracted.strip()) > 200:
        return extracted.strip()

    # Fallback parser
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")

    # Normalize unicode quotes/dashes if present
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")

    # Fix some common smashed spacing issues
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(@[a-zA-Z_]+)", r"\1 \2", text)
    text = re.sub(r"([a-z])(\.)([A-Z])", r"\1. \3", text)

    # Fix missing spaces around punctuation/brackets in common cases
    text = re.sub(r"([,:;])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\()", r"\1 \2", text)
    text = re.sub(r"(\))([A-Za-z])", r"\1 \2", text)

    # Normalize whitespace early
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    junk_patterns = [
        r"^Connect these docs to",
        r"^Copy page$",
        r"^Was this page helpful\??$",
        r"^Search$",
        r"^Skip to content$",
        r"^Table of contents$",
        r"^On this page$",
        r"^API reference$",
        r"^Previous$",
        r"^Next$",
        r"^Edit this page$",
        r"^Open in GitHub$",
        r"^Back to top$",
        r"^Loading\.\.\.$",
        r"^Section titled .*$",
        r"^Was this helpful\??$",
        r"^Share$",
        r"^Menu$",
    ]

    cleaned_lines = []
    prev_line = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in junk_patterns):
            continue

        # Drop lines that are mostly symbols/noise
        alpha_chars = sum(ch.isalpha() for ch in line)
        if alpha_chars == 0 and len(line) < 40:
            continue

        # Drop very short UI-ish fragments, but keep meaningful technical terms
        if len(line) < 5 and line.lower() not in {"rag", "api", "llm"}:
            continue

        # Remove duplicate consecutive lines
        if prev_line and line.lower() == prev_line.lower():
            continue

        cleaned_lines.append(line)
        prev_line = line

    text = "\n".join(cleaned_lines)

    # Remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix spacing before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Remove spaces just inside brackets/parentheses
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()