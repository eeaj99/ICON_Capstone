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
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

# API KEYS
api_key = os.getenv("GROQ_API_KEY")

# CHAT MODEL INITIALIZATION
model = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
print("Chat model initialized successfully.")

# HUGGINGFACE EMBEDDINGS 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("Embeddings initialized successfully.")

# PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are a responsible and professional assistant designed to support Adverse Event (AE) reporting for pharmaceutical products. Your task is to extract only the most relevant sentences from the provided context to answer the user's question.
     
        Disallowed answering patterns:
        <think> We might consider this...
        <reasoning> Because X implies Y...

        Instructions:
        1. Base your response strictly on the provided context. Do not use external knowledge.
        2. Ignore any sections labeled "Table of Contents" and "Index". Do not extract content from these sections. Focus only on the main body of the document that directly answers the question.
        3. Do not include any <think>, <thought>, or internal reasoning tags in your response. Only return direct answers from the provided context.
        4. Do not provide reasoning, explanation, or commentary.
        5. Ensure selected sentences are semantically and contextually aligned with the question.
        6. Review the entire context carefully to find the most relevant matches.
        7. Your response must be clear, in-detail, properly formatted (bullet/numbered wise) and strictly limited to the matching content. 
        8. If the question is not related to Adverse Event reporting, respond with: "This question is not related to Adverse Event reporting. Please ask a relevant question."
        9. Return 'NO RELEVANT CONTEXT FOUND' if nothing applies. 
        10.. Don't mention explanation or reasoning in your response. 

        Do not improvise. Follow the instructions exactly.
    """),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Initialize memory for the conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Extract text from PDF
def extract_pdf_text(pdf_path):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)

# Vectorization
def vectorize_extracted_content(text_content):
    docs = [Document(page_content=text_content)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# Main execution flow
print("Starting the document processing pipeline...")
# Paths 
pdf_path = "doc1.pdf"
extracted_text = extract_pdf_text(pdf_path)
print("PDF text extracted successfully.")

# Save extracted text to a file
# extracted_text_path = "extracted_text.txt"
# with open(extracted_text_path, "w", encoding="utf-8") as f:
#     f.write(extracted_text)

# Vectorization
vectorstore = vectorize_extracted_content(extracted_text)
print("Content vectorized successfully.")

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", k=5)
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
    if user_input.strip() == "":
        print("Please enter a valid question.")
        continue

    # retrieved_docs = retriever.get_relevant_documents(user_input)
    # print("\n--- Retrieved Chunks ---")
    # for i, doc in enumerate(retrieved_docs):
    #     print(f"[Chunk {i+1}]\n{doc.page_content}\n")
    # print("\n--- End of Retrieved Chunks ---\n")

    # Invoke the QA chain with the user's question
    response = qa_chain.invoke({"question": user_input})
    print("Bot:", response["answer"])

print("Document processing pipeline completed successfully.")


