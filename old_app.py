import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize model and embeddings
model = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are a responsible and professional assistant designed to support Adverse Event (AE) reporting for pharmaceutical products. Your task is to extract only the most relevant sentences from the provided context to answer the user's question.

        Disallowed answering patterns:
        <think> We might consider this...
        <reasoning> Because X implies Y...
     
        Instructions:
        1. Base your response strictly on the provided context. Do not use external knowledge.
        2. Ignore any sections labeled "Table of Contents" and "Index". Do not extract content from these sections.
        3. Do not include any <think>, <thought>, or internal reasoning tags.
    """),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# PDF text extraction
def extract_text_from_pdfs(uploaded_files):
    full_text = []
    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
    return "\n".join(full_text)

# Vectorization
def vectorize_text(text):
    docs = [Document(page_content=text)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# Streamlit UI
st.set_page_config(page_title="AE RAG Assistant", layout="wide")
st.title("ADE Documentation Assistant")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Reading and vectorizing documents..."):
            text = extract_text_from_pdfs(uploaded_files)
            vectorstore = vectorize_text(text)
            retriever = vectorstore.as_retriever(search_type="similarity", k=5)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=model,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                output_key="answer"
            )
            st.session_state.qa_chain = qa_chain
            st.success("Ready to chat with your documents.")

# Chat Interface
if "qa_chain" in st.session_state:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question about your documents")

    if user_input and "qa_chain" in st.session_state:
        with st.spinner("Fetching response..."):
            response = st.session_state.qa_chain.invoke({"question": user_input})
            answer = response["answer"]
            source_chunks = response["source_documents"]

            # Save to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

