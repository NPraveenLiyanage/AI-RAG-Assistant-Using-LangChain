# Quest Analytics RAG Assistant

Quest Analytics RAG Assistant is a lightweight research copiloting app that lets you upload PDFs and ask natural-language questions with grounded, source-aware responses. It combines a clean Streamlit chat UI with a modular LangChain pipeline for document loading, chunking, embedding, vector search, and response generation. The project is designed to be easy to run locally, simple to extend, and clear to study as a reference implementation of end‑to‑end RAG.

## Highlights
- Streamlit chat interface with PDF upload
- Modular LangChain pipeline (load → split → embed → vector store → retriever → QA)
- Local-first setup with Ollama for free, fast experimentation
- Optional watsonx.ai support for hosted embeddings and LLMs
- Clean structure for quick customization and learning

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. If using free local path, install and run Ollama (https://ollama.com) with models `llama3` and `mxbai-embed-large` pulled; keep `USE_OLLAMA` unset or true.
4. If using watsonx.ai, set env vars `USE_OLLAMA=false`, `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID`.
5. Place your PDF at `data/research.pdf` (or update the path). Optional: replace `data/notes.txt`.
6. Run the demo: `python rag_pipeline.py`.

Open `rag_pipeline.py` and capture these views:
- Task 1 loader code → save as `pdf_loader.png`.
- Task 2 splitter code → `code_splitter.png`.
- Task 3 embeddings code → `embedding.png`.
- Task 4 Chroma setup → `vectordb.png`.
- Task 5 retriever code → `retriever.png`.

## Notes
- The code is defensive: it skips local loaders if files are absent so you can run with only the web source at first.
- Free path: LLM `llama3` + embeddings `mxbai-embed-large` via local Ollama.
- Watsonx path (optional): embeddings `ibm/slate-125m-english-rtrvr`; LLM `mistralai/mixtral-8x7b-instruct-v01`.
