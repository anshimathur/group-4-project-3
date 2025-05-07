# 📚 NLP-Powered Q&A Over Class Materials

## 🚀 Project Overview

A natural language interface that enables users to query class materials — including PowerPoints, PDFs, transcripts, and Jupyter notebooks — and receive relevant, accurate responses. The system also returns the source or module in which the answer was originally taught.

This project was developed as part of our final class assignment to demonstrate practical implementation of NLP, embeddings, and retrieval-augmented models using course data.

---

## 🎯 Goals

- Allow users to ask natural language questions about class content
- Return contextual answers pulled directly from course materials
- Display the original module or document source for reference
- Compare local and cloud-based approaches for speed, accuracy, and privacy
- Compare vector-only retrieval vs. full RAG (retrieval-augmented generation) pipelines

---

## 📂 Data Collection & Preprocessing

We plan to ingest a wide range of course content, including:

- PDF lecture slides and reading materials
- Zoom meeting transcripts
- Jupyter notebook code and markdown cells (potentially)
- Class activity files from VSCode

### Preprocessing Steps:

- Converted files into plain text with LangChain loaders
- Applied recursive text splitting for token-length optimization
- Embedded using `sentence-transformers` models
- Indexed with FAISS for efficient vector search

---

## 🤖 Model Architecture

We're currently testing two deployment configurations:

### 🔁 Vector-Based Retrieval

- Uses FAISS for semantic search
- Returns top-k documents or chunks as plain responses

### 🔍 RAG-Based QA Pipeline

- Integrates document retrieval with LLM generation
- Maintains limited conversation history to prevent context overflow
- Uses either a local LLM or cloud-based models (e.g., Gemini)
- Provides more natural, summarized answers

---

## 🧪 Model Optimization

- Iteratively tuned embedding model (e.g., `all-MiniLM-L6-v2`) and chunk sizes
- Compared performance across retrieval top-k values and LLM prompt settings
- Logged results and timings to CSV for reproducibility and analysis

---

## 🔐 Privacy & Speed Notes

We are testing **both local and cloud-based deployment** options:

### Local Mode
- Fully private — no external calls
- Powered by SentenceTransformers + FAISS
- Can integrate with local LLMs (e.g., GPT4All, LLaMA)

### Cloud Mode
- Connects to Gemini API (or other LLM APIs)
- Offers superior generation quality
- Slight latency due to API call time

---

## 🖥️ Presentation Summary

- **Executive Summary**: We built a smart Q&A system over our course materials using modern NLP tools
- **Data Collection**: Aggregated and cleaned PDFs, notebooks, and transcripts
- **Approach**: Compared vector retrieval vs RAG; tested local vs cloud hosting
- **Next Steps**: Add support for uploading new docs, evaluate large-scale accuracy, test with new LLMs
- **Conclusion**: Shows how AI can make studying faster, easier, and more interactive

---

## 👩‍💻 Technologies Used

- Python
- Jupyter Notebooks
- LangChain
- FAISS
- SentenceTransformers
- Gemini API (optional)
- Streamlit (optional UI layer)
- GitHub

---

## 📌 Authors

- Anshi Mathur, Chad Bradforb, Peyton Lambourne, James Segovia

