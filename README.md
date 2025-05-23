# 📚 AI Course Tutor — NLP-Powered Q\&A over Class Materials

## 🚀 Project Overview

A natural language interface that enables users to query class materials — including PowerPoints, PDFs, transcripts, and Jupyter notebooks — and receive relevant, accurate responses. The system also returns the source or module in which the answer was originally taught.

This project was developed as part of our final class assignment to demonstrate practical implementation of NLP, embeddings, and retrieval-augmented models using course data.

---

## 🎯 Goals

* Allow users to ask natural language questions about class content
* Return contextual answers pulled directly from course materials
* Display the original module or document source for reference
* Compare local and cloud-based approaches for speed, accuracy, and privacy
* Compare vector-only retrieval vs. full RAG (retrieval-augmented generation) pipelines

---

## 📂 Data Collection & Preprocessing

We ingested a wide range of course content, including:

* PDF lecture slides and reading materials
* Zoom meeting transcripts
* Jupyter notebooks and Python scripts
* Class activity files from VSCode

### Preprocessing Steps

* Converted files into plain text using custom readers (including `.ipynb`, `.py`, `.md`, `.pdf`, `.txt`)
* Applied recursive and language-aware text splitting for semantic chunking
* Embedded course content through Google Gemini's 'models/embedding-001' via GoogleGenerativeAIEmbeddings.
* Indexed documents with FAISS, tagging each chunk with module, day, and source info

---

## 🧰 Application Setup

A full walkthrough is available in [`SETUP_GUIDE.md`](./SETUP_GUIDE.md), including Conda environment setup, dependency installation, and index generation.

### Highlights:

* Uses Streamlit as a lightweight UI
* Integrates FAISS for semantic search
* Gemini API is required (set via `.env`)
* Automatically builds a manifest of all supported content types (`.pdf`, `.py`, `.ipynb`, `.md`, `.txt`)

### Run the App

```bash
conda activate tutor_env
python -m streamlit run Chat.py
```

On first launch, the app builds a FAISS vector index — this may take several minutes.

---

## 🤖 Model Architecture

### 🔁 Vector-Based Retrieval

* Uses FAISS for semantic document matching
* Returns top-k relevant chunks with source metadata

### 🔍 RAG-Based QA Pipeline

* Integrates LLM generation (Gemini) with document retrieval
* Summarizes or elaborates answers based on context
* Preserves recent chat history for improved coherence

---

## 🧪 Optimization & Evaluation

* Tuned chunk sizes and overlap for optimal embedding context
* Evaluated performance in both local and cloud settings

---

## 🔐 Privacy vs. Speed

### Local Model

* No API calls; embeddings and search are performed locally
* Privacy-preserving, ideal for sensitive material
* Con: Very slow message output due to available local resources.

### Cloud Model

* Uses Gemini API for natural language generation
* Richer responses
* Con: Requires an API key and has limited use.

We opted to build the app using a Cloud Model.

---

## 🖥️ Presentation Summary

* **Executive Summary**: We built a smart Q\&A system over course materials using FAISS + Gemini
* **Data Collection**: Aggregated PDFs, notebooks, transcripts
* **Architecture**: Indexed with FAISS, queried via Streamlit, RAG-capable
* **Next Steps**: Add document upload, improve UI, test with alternate LLMs
* [Presentation Link](https://www.canva.com/design/DAGm7CUO0nY/NkyQaZm8ULOkdbpqUbvh5g/edit?utm_content=DAGm7CUO0nY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 👩‍💻 Technologies Used

* Python
* Jupyter Notebooks
* LangChain (preliminary)
* FAISS
* Google Gemini API
* Streamlit
* GitHub

---

## 📌 Authors

* Anshi Mathur
* Chad Bradford
* Peyton Lambourne
* James Segovia
