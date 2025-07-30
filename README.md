ADE Document Assistant

The ADE Document Assistant is a document-aware conversational AI application developed to support Adverse Event (AE) reporting for pharmaceutical and medical device products. It leverages Retrieval-Augmented Generation (RAG) techniques using OpenAI’s LLMs and LangChain to provide precise, contextually grounded answers based on internal FDA documentation.

Features

RAG-powered QA : Accurate question answering over FDA-issued PDFs using similarity-based retrieval.
Conversational Memory : Context-aware conversation handling using LangChain's `ConversationalRetrievalChain`.
Modular Prompting : Custom system prompts enforce strict relevance filtering and document fidelity.
Dynamic Document Loading : Upload and vectorize PDFs categorized by FDA centers.
Interactive Chat Interface : User-friendly UI built with Streamlit.

Project Structure

ICON_Capstone/
│
├── app.py                   # Streamlit front-end application
├── config.py                # Environment loader and document config parser
├── prompts.py               # System prompt template used by the LLM
├── rag_engine.py            # Core RAG logic: extract, chunk, embed, retrieve
├── document_map.json        # Maps FDA centers to document paths
├── requirements.txt         # Python dependencies
├── .env                     # API keys and env variables
├── style.css                # Custom styles for the Streamlit interface
└── documents/               # Folder to store all referenced PDFs


Tech Stack

Python 3.10+
Streamlit  — for UI
LangChain — LLM orchestration & memory
FAISS — fast vector search
OpenAI GPT-4o-mini — primary language model
PDFPlumber — PDF parsing
dotenv— environment config


Setup Instructions

1. Clone the Repository
git clone https://gitlab.com/your-org/ICON_Capstone.git
cd ICON_Capstone

2. Install Dependencies
pip install -r requirements.txt

3. Set Up Environment Variables
Update the `.env` file with your OpenAI API key:

5. Run the App
streamlit run app.py
