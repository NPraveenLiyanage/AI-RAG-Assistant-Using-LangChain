"""
Quest Analytics RAG assistant demo using LangChain.
Default path runs via Hugging Face Router (Zephyr-7B beta chat) with MiniLM embeddings.
This script follows the six requested tasks end-to-end.
"""
from pathlib import Path
import os
from typing import List, Optional

from dotenv import load_dotenv

# Load .env early, then set user agent headers before other network-using imports
load_dotenv()
default_agent = os.getenv("USER_AGENT", "QuestRagDemo/1.0")
os.environ["USER_AGENT"] = default_agent
os.environ.setdefault("HUGGINGFACEHUB_USER_AGENT", default_agent)

import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


def _require_hf_token() -> None:
    """Make sure a Hugging Face Hub token is available before API calls."""
    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        raise EnvironmentError(
            "Set HF_TOKEN (preferred) or HUGGINGFACEHUB_API_TOKEN in the environment or .env file to use Hugging Face Hub."
        )


# --- Task 1: load documents from multiple sources --------------------------------
def load_documents(
    pdf_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    text_path: Optional[Path] = None,
    urls: Optional[List[str]] = None,
) -> List[Document]:
    data_dir = Path("data")

    resolved_pdf = Path(pdf_path) if pdf_path else None
    resolved_csv = Path(csv_path) if csv_path else None
    resolved_txt = Path(text_path) if text_path else data_dir / "notes.txt"

    loaders = []
    if resolved_pdf and resolved_pdf.exists():
        loaders.append(PyPDFLoader(str(resolved_pdf)))
    if resolved_csv and resolved_csv.exists():
        loaders.append(CSVLoader(str(resolved_csv)))
    if resolved_txt and resolved_txt.exists():
        loaders.append(TextLoader(str(resolved_txt), encoding="utf-8"))
    if urls:
        loaders.append(WebBaseLoader(urls))

    docs: List[Document] = []
    for loader in loaders:
        docs.extend(loader.load())
    if not docs:
        raise FileNotFoundError(
            "No documents found. Provide at least one valid PDF, CSV, text file, or URL."
        )
    return docs


# --- Task 2: apply text splitting -------------------------------------------------
def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


# --- Task 3: embed documents ------------------------------------------------------
def build_embeddings():
    _require_hf_token()
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


# --- Task 4: create and configure vector DB (Chroma) -----------------------------
def build_vectorstore(splits: List[Document], embed_model) -> Chroma:
    persist_dir = Path("vectorstore/chroma")
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embed_model,
        persist_directory=str(persist_dir),
    )
    return vectordb


# --- Task 5: develop retriever ----------------------------------------------------
def build_retriever(vectordb: Chroma):
    return vectordb.as_retriever(search_type="mmr", 
                                 search_kwargs={"k": 4, "fetch_k": 12}
                                 )


# --- Task 6: construct QA bot -----------------------------------------------------
def build_qa_bot(retriever):
    _require_hf_token()
    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        temperature=0.2,
        max_tokens=256,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise research assistant. Use the provided context to answer."),
            ("user", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def demo(query: str = "What this paper is talking about?", pdf_path: Optional[Path] = None):
    docs = load_documents(pdf_path)
    if not docs:
        raise RuntimeError("No documents loaded. Ensure the lab PDF is present.")
    splits = split_documents(docs)
    embed_model = build_embeddings()
    vectordb = build_vectorstore(splits, embed_model)
    retriever = build_retriever(vectordb)
    qa_chain = build_qa_bot(retriever)

    answer = qa_chain.invoke(query)
    sources = retriever.invoke(query)

    print("\nAnswer:\n", answer)
    print("\nSources used:")
    for doc in sources:
        print("-", doc.metadata.get("source"))


if __name__ == "__main__":
    demo()
