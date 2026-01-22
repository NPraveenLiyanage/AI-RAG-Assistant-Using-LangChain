import os
import tempfile
from pathlib import Path

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag_pipeline import (
    load_documents,
    split_documents,
    build_embeddings,
    build_vectorstore,
    build_retriever,
    build_qa_bot,
)

st.set_page_config(page_title="Quest Analytics", layout="centered", initial_sidebar_state="collapsed")

# Claude-style clean CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: #1a1a1a; }
header[data-testid="stHeader"] { display: none; }
footer { display: none; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }

.block-container { padding: 2rem 1rem 8rem 1rem; max-width: 760px; }

.hero { text-align: center; padding: 3rem 0 2rem 0; }
.hero h1 { font-size: 2rem; font-weight: 600; color: #f5f5f5; margin: 0; }
.hero p { font-size: 0.95rem; color: #888; margin-top: 0.5rem; }

.msg-row { display: flex; margin: 1rem 0; }
.msg-user { margin-left: auto; background: #303030; color: #f0f0f0; padding: 0.85rem 1.1rem; border-radius: 1.25rem 1.25rem 0.25rem 1.25rem; max-width: 80%; }
.msg-assistant { background: #262626; color: #e8e8e8; padding: 0.85rem 1.1rem; border-radius: 1.25rem 1.25rem 1.25rem 0.25rem; max-width: 80%; }

.input-dock { position: fixed; bottom: 0; left: 0; right: 0; background: linear-gradient(transparent, #1a1a1a 20%); padding: 1rem 1rem 1.5rem 1rem; }
.input-box { max-width: 720px; margin: 0 auto; background: #262626; border: 1px solid #3a3a3a; border-radius: 1.5rem; display: flex; align-items: center; padding: 0.5rem 0.75rem; gap: 0.5rem; }
.input-box:focus-within { border-color: #555; }

div[data-testid="stTextInput"] > div { background: transparent !important; }
div[data-testid="stTextInput"] input { background: transparent !important; border: none !important; color: #f0f0f0 !important; font-size: 0.95rem !important; padding: 0.6rem 0 !important; }
div[data-testid="stTextInput"] input::placeholder { color: #777 !important; }
div[data-testid="stTextInput"] input:focus { box-shadow: none !important; }

div[data-testid="stFileUploader"] { width: 40px; min-width: 40px; }
div[data-testid="stFileUploader"] > label { display: none !important; }
div[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; padding: 0 !important; min-height: 40px !important; }
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { padding: 0 !important; border: none !important; background: transparent !important; min-height: 40px !important; }
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }

div[data-testid="stFileUploader"] button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #262626 !important;
    border: 1px solid #3a3a3a !important;
    width: 40px !important;
    height: 40px !important;
    border-radius: 999px !important;
    padding: 0 !important;
    cursor: pointer !important;
    color: transparent !important;
    position: relative !important;
    overflow: hidden !important;
}
div[data-testid="stFileUploader"] button::after {
    content: "";
    position: absolute;
    inset: 0;
    background: center / 18px 18px no-repeat;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23cfcfcf' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21.44 11.05l-8.49 8.49a6 6 0 01-8.49-8.49l8.49-8.49a4 4 0 015.66 5.66l-8.49 8.49a2 2 0 01-2.83-2.83l7.78-7.78'/></svg>");
}

button[data-testid="stBaseButton-primary"] { background: #4a4a4a !important; border: none !important; border-radius: 999px !important; padding: 0.55rem 1.1rem !important; font-weight: 500 !important; color: #fff !important; font-size: 0.9rem !important; }
button[data-testid="stBaseButton-primary"]:hover { background: #5a5a5a !important; }

.file-chip { display: inline-flex; align-items: center; gap: 0.4rem; background: #2e2e2e; color: #bbb; font-size: 0.8rem; padding: 0.35rem 0.7rem; border-radius: 0.75rem; margin-top: 0.5rem; }

</style>
""",
    unsafe_allow_html=True,
)

# Token check (local env or Streamlit secrets)
hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or st.secrets.get("HF_TOKEN")
    or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
)
if not hf_token:
    st.error(
        "Missing Hugging Face token. Add HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) "
        "in Streamlit Secrets or as an environment variable."
    )
    st.stop()


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
        model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        temperature=0.2,
        max_tokens=256,
    )


@st.cache_resource(show_spinner=False)
def _build_rag(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        docs = load_documents(tmp_path)
        splits = split_documents(docs)
        embed_model = build_embeddings()
        vectordb = build_vectorstore(splits, embed_model)
        retriever = build_retriever(vectordb)
        qa_chain = build_qa_bot(retriever)
        return qa_chain, retriever
    finally:
        tmp_path.unlink(missing_ok=True)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.retriever = None

# Hero
st.markdown("<div class='hero'><h1>Quest Analytics</h1><p>AI Research Assistant</p></div>", unsafe_allow_html=True)

# Messages
for msg in st.session_state.messages:
    cls = "msg-user" if msg["role"] == "user" else "msg-assistant"
    st.markdown(f"<div class='msg-row'><div class='{cls}'>{msg['content']}</div></div>", unsafe_allow_html=True)


def _submit_prompt() -> None:
    prompt = st.session_state.get("prompt", "").strip()
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    if st.session_state.rag_chain:
        answer = st.session_state.rag_chain.invoke(prompt)
        reply = answer
    else:
        llm = _get_llm()
        resp = llm.invoke([
            SystemMessage(content="You are a concise research assistant."),
            HumanMessage(content=prompt),
        ])
        reply = resp.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.prompt = ""
    st.session_state.pending_send = True

# Input dock
st.markdown("<div class='input-dock'><div class='input-box'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([0.06, 0.78, 0.16])
with c1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed", key="pdf")
with c2:
    st.text_input(
        "Question",
        placeholder="Ask anything...",
        label_visibility="collapsed",
        key="prompt",
        on_change=_submit_prompt,
    )
with c3:
    send = st.button("Send", type="primary", key="send")
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("pending_send"):
    st.session_state.pending_send = False
    st.rerun()

if uploaded_file:
    st.markdown(f"<div class='file-chip'>📄 {uploaded_file.name}</div>", unsafe_allow_html=True)
    if st.session_state.rag_chain is None:
        qa, ret = _build_rag(uploaded_file.read())
        st.session_state.rag_chain = qa
        st.session_state.retriever = ret

st.markdown("</div>", unsafe_allow_html=True)

# Handle send
if send:
    _submit_prompt()
