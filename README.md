# Quest Analytics RAG Assistant

This repo contains a minimal LangChain RAG pipeline for Quest Analytics. The code in `rag_pipeline.py` aligns with the six requested tasks (load docs, split, embed, vector DB, retriever, QA bot). Default path uses free local Ollama (llama3 + mxbai-embed-large). watsonx.ai remains available when you set `USE_OLLAMA=false` and provide credentials.

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. If using free local path, install and run Ollama (https://ollama.com) with models `llama3` and `mxbai-embed-large` pulled; keep `USE_OLLAMA` unset or true.
4. If using watsonx.ai, set env vars `USE_OLLAMA=false`, `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID`.
5. Place your PDF at `data/research.pdf` (or update the path). Optional: replace `data/notes.txt`.
6. Run the demo: `python rag_pipeline.py`.

## Screenshots to deliver
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
