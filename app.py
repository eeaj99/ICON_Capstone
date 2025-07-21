import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from config import load_api_key, load_document_config
from prompts import get_prompt_template
from rag_engine import extract_text_from_pdfs, vectorize_text, create_qa_chain

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# App config
st.set_page_config(page_title="ADE Document Assistant", layout="wide")
# st.subheader("ADE Document Assistant")

# Load API key and models
api_key = load_api_key()
model = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
prompt_template = get_prompt_template()
doc_config = load_document_config()


st.markdown("""
<div class="fixed-header">
    <div class="header-title">ADE Document Assistant</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<p style='margin-bottom: 0.2rem;'>Select FDA Organization</p>", unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns([4, 3])
    with col1:
        fda_organizations = list(doc_config.keys())
        selected_org = st.selectbox("", fda_organizations, label_visibility="collapsed")
    with col2:
        button_col, msg_col = st.columns([1, 2])
        with button_col:
            load_clicked = st.button("Load Documents")
        with msg_col:
            message_placeholder = st.empty()

if load_clicked:
    with message_placeholder:
        with st.spinner(""):
            org_docs = doc_config[selected_org]  # dict of {doc_name: doc_path}
            for doc_name, doc_path in org_docs.items():
                if doc_name not in st.session_state:
                    with open(doc_path, "rb") as f:
                        text = extract_text_from_pdfs([f])
                    vectorstore = vectorize_text(text, embeddings)
                    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
                    st.session_state[doc_name] = create_qa_chain(model, retriever, prompt_template)

            # Set the first document in the org as active QA chain
            first_doc_name = next(iter(org_docs))
            st.session_state.qa_chain = st.session_state[first_doc_name]
        
        # Success message replaces the spinner in the same location
        st.text(f"Loaded {len(org_docs)} documents.")

# Chat Interface
if "qa_chain" in st.session_state:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question")
    if user_input:
        # with st.spinner("Fetching response..."):
        response = st.session_state.qa_chain.invoke({"question": user_input})
        answer = response["answer"]
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
