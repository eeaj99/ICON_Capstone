import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from config import load_api_key, load_document_config
from prompts import get_prompt_template
from rag_engine import extract_text_from_pdfs, vectorize_text, create_qa_chain

# App config
st.set_page_config(page_title="ADE Document Assistant", layout="wide")
# st.header("ADE Document Assistant")
st.markdown("""
    <style>
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #ffffff;
            padding: 12px;
            font-size: 20px;
            text-align: left;
            margin-left: 210px;
            z-index: 9999;
            border-bottom: 1px solid #ccc;
        }
        .main {
            padding-top: 70px;  
        }
    </style>
    <div class="fixed-header">ADE Document Assistant</div>
""", unsafe_allow_html=True)



# Load API key and models
api_key = load_api_key()
model = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
prompt_template = get_prompt_template()
doc_config = load_document_config()

# Sidebar for file upload
# with st.sidebar:
    # st.subheader("Upload Documents")
    # uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    # if uploaded_files and st.button("Process Documents"):
    #     with st.spinner("Reading and vectorizing documents..."):
    #         text = extract_text_from_pdfs(uploaded_files)
    #         vectorstore = vectorize_text(text, embeddings)
    #         retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    #         st.session_state.qa_chain = create_qa_chain(model, retriever, prompt_template)
    #         st.success("Ready to chat with your documents.")
    # Sidebar for document selection
with st.sidebar:
    # st.subheader("Select Document")
    doc_options = []
    doc_map = {}
    for org_docs in doc_config.values():
        for doc_name, doc_path in org_docs.items():
            doc_options.append(doc_name)
            doc_map[doc_name] = doc_path

    selected_doc = st.radio(label="Select Document", options=doc_options)
    selected_path = doc_map[selected_doc]
    selected_key = selected_doc 

    if st.button("Load Document"):
        if selected_key not in st.session_state:
            with st.spinner("Loading and vectorizing document..."):
                with open(selected_path, "rb") as f:
                    text = extract_text_from_pdfs([f])
                vectorstore = vectorize_text(text, embeddings)
                retriever = vectorstore.as_retriever(search_type="similarity", k=5)
                st.session_state[selected_key] = create_qa_chain(model, retriever, prompt_template)
        st.session_state.qa_chain = st.session_state[selected_key]
        st.success(f"{selected_doc} loaded successfully.")





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
