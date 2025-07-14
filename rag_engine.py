import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def extract_text_from_pdfs(uploaded_files):
    full_text = []
    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
    return "\n".join(full_text)

def vectorize_text(text, embeddings):
    docs = [Document(page_content=text)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

def create_qa_chain(model, retriever, prompt_template):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        output_key="answer"
    )
