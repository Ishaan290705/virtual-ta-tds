# TDS Virtual TA – IITM "Tools in Data Science" Question Answering API
Welcome to the TDS Virtual TA project! This is a powerful, modern API designed to answer questions for the IIT Madras "Tools in Data Science" (TDS) course. It combines web scraping, LLM-powered semantic embeddings, and similarity search to deliver smart, context-aware, and course-aligned answers.

---

## 🚀 Features
- FastAPI server for interactive API endpoints
- Semantic similarity search over:
  - Historical student Q&A (from the IITM Discourse forum)
  - Official TDS course content (scraped automatically!)
- Leverages OpenAI's powerful embeddings and models (GPT-4o, etc.)
- Supports screenshot/image question OCR and text extraction
- Smart, context-driven answer construction using few-shot LLM prompting
- Links to the official TDS content or historical posts for transparency
- Easily extensible and well-structured for research, bots, or education tools

---


## ⚡️ How It Works
### 1. Data Collection
- Crawls the course's Discourse forum for hundreds of Q&A pairs.
- Scrapes and processes the official TDS course handbook/notes.
- Applies OCR extraction to images/screenshots in posts if present.
### 2. Semantic Embedding
- Uses OpenAI models to convert questions/answers and course content into vector embeddings.
- Caches and manages all embeddings for efficiency.
### 3. Question Answering Flow
- Accepts user questions (and optional screenshots).
- Computes embedding for the input.
- <b>Tries to match with:</b>
  - Historical forum Q&A (highest semantic similarity)
  - Official TDS course content (passage-level similarity)
  - If no relevant match, applies custom rules for topic lookup using course metadata.
### 4. LLM-Powered Answer Synthesis
- LLM is prompted using the most-relevant context(s) and generates a JSON-structured answer.
- Output always links to the most-relevant sources.

---



