import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize model and embeddings
model = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
        4. Return 'NO RELEVANT CONTEXT FOUND' if nothing applies.
        5. If the question is not related to AE reporting, respond with: "This question is not related to Adverse Event reporting. Please ask a relevant question."
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
    chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# Streamlit UI
st.set_page_config(page_title="AE RAG Assistant", layout="wide")
st.title("Adverse Event RAG Assistant")

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
    st.subheader("Ask a question")
    user_question = st.text_input("Enter your question")

    if user_question:
        with st.spinner("Fetching response..."):
            response = st.session_state.qa_chain.invoke({"question": user_question})
            st.markdown("### Answer")
            st.write(response["answer"])

            with st.expander("Retrieved Context"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(doc.page_content)
