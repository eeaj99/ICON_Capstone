# IMPORTS
import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import PyPDF2

# API KEYS
# os.environ['GROQ_API_KEY'] = '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'
os.environ['GROQ_API_KEY'] = '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'

# CHAT MODEL INITIALIZATION
model = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
print("Chat model initialized successfully.")

# HUGGINGFACE EMBEDDINGS 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("Embeddings initialized successfully.")

# PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Reminder:
        DO NOT PRINT WHAT YOU THINK
        DO NOT GIVE REASON FOR YOUR ANSWERS AT ALL.
        Understand the document and answer the question based on the content provided.
        You are a helpful assistant that extracts relevant sentences from the provided context to answer the question.
     
        Instructions:
        Your response must consist solely of exact sentences from the provided context. Do not generate new sentences, rephrase, summarize, or add external information.
        Extract at max 3 sentences that best matches the meaning of the given definition from the provided relevant content.
        If no relevant content exists, respond with: "NO RELEVANT CONTEXT FOUND", without any explanation. DO NOT MAKE ANS OF YOUR OWN.
        Ensure that the extracted text from the relevant content is contextually and semantically aligning with the definition. Do not infer meaning beyond what is explicitly stated in the context.
     
        Reminder:
        DO NOT GIVE REASON FOR YOUR ANSWERS AT ALL.
        Only return sentences exactly as they appear in the provided context. Do not summarize, reword, or infer additional meaning.
    """),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Initialize memory for the conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Extract text from PDF
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return extracted_text

# Vectorization
def vectorize_extracted_content(text_content):
    docs = [Document(page_content=text_content)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore



# Main execution flow
print("Starting the document processing pipeline...")
# Paths
pdf_path = "D:\\SEM 3\\Capstone\\RAG\\doc1.pdf"
extracted_text = extract_pdf_text(pdf_path)
print("PDF text extracted successfully.")
print(f"Extracted text length: {len(extracted_text)} characters")
# print(f"Extracted text preview: {extracted_text[:500]}...")  # Preview first 500 characters

# Vectorization
vectorstore = vectorize_extracted_content(extracted_text)
print("Content vectorized successfully.")

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", k=3)
print("Retriever created successfully.")

# Create a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    output_key="answer"
)

# Chat loop
print("\nChatbot is ready! Type your question (or type 'exit' to quit):\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = qa_chain.invoke({"question": user_input})
    print("Bot:", response["answer"])

print("Document processing pipeline completed successfully.")


