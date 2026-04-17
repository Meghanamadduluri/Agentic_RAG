Agentic RAG for Technical Decision Support
A specialized Agentic Retrieval-Augmented Generation (RAG) system designed to provide structured, deep-dive comparisons between complex AI frameworks like LangChain and LlamaIndex.

Unlike standard "single-pass" RAG systems that often suffer from shallow technical coverage, this system utilizes a Planner-Executor architecture to decompose queries into multi-step research tasks, improving technical aspect coverage by ~40%.

The Core Problem
Standard RAG systems typically perform a single "top-k" retrieval. When asked a complex comparison question (e.g., "Which framework is better for high-concurrency memory management?"), the results are often:
Biased: One framework may dominate the retrieved context.
Shallow: Niche technical details (like specific persistence utilities) are often missed.
Noisy: Context-stuffing leads to hallucinations or "lost in the middle" phenomena.

System Architecture
This project implements an Agentic Research Loop that mimics the workflow of a human technical consultant.
1. The Planner (Orchestration)The "Brain" of the system. It analyzes the raw user query and decomposes it into:
Entities: Identifying which frameworks to research.
Aspects: Identifying specific technical sub-topics (e.g., streaming, storage, evaluation).

2. The Decomposed Executor (Iterative Retrieval)
Instead of one broad search, the agent executes 10-16 targeted retrieval calls. It uses Gemini Embeddings and ChromaDB with strict metadata filtering to pull high-signal evidence for every specific entity-aspect pair.

3. Synthesis & Grounding
The final generation layer synthesizes the partitioned research into a structured JSON comparison, ensuring every claim is grounded in the retrieved technical documentation.

Performance Metrics
Based on a benchmarking suite of decision-support queries, the agentic approach showed significant improvements over the baseline:

Metric                     Baseline RAG             Agentic RAG
Retrieval Calls           1-2 per query              10-16 per query 
Aspect Coverage            ~50%                     Near-Complete 
Recommendation Depth       General/Surface          Technical/Specific 
Latency                    ~5-10s                     ~45-60s 

Tech Stack
LLMs: Gemini 1.5 Flash & Pro (via Google GenAI SDK) 
Vector Store: ChromaDB 
Backend: Python
Workflow: AI-assisted development (Cursor, Gemini)  

Installation & Usage

Clone the repo:Bashgit clone https://github.com/yourusername/agentic-rag-comparison.git

Setup environment:Create a .env file and add your GOOGLE_API_KEY.

Run the system:
Bash
python src/baseline_compare.py for baseline RAG responses
python src/agentic_compare.py for validating agentic responses