# 🎓 RAG-Powered Admission Query System

This project is a capstone system that allows students or users to ask admission-related queries and receive intelligent, context-aware responses using state-of-the-art LLMs and embeddings. It supports Retrieval-Augmented Generation (RAG), Supabase-based authentication, and LangChain integration to ensure secure and efficient interaction.

---

## 🧠 Features

- ✅ RAG-based query resolution for student admission FAQs
- ✅ Authentication system using Supabase (Sign up & Login with JWT)
- ✅ Fast and relevant responses using LangChain + FAISS vector database
- ✅ HuggingFace LLM integration (Mistral-7B)
- ✅ Custom prompt templating for more accurate and safe responses

### Clone the Repository

## Create and Activate Virtual Environment

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Add Environment Variables
In your .env

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
HF_TOKEN=your_huggingface_access_token
```
## Generating the Vector DB

** run the file named: ./rag_test/create_vector_db.py
This will generate:
```
vectorstore/db_faiss/
├── index.faiss
└── index.pkl
```
** ⚠️ Make sure index.faiss and index.pkl are generated before running the main app.

** you can also test the llm+rag without fast-api by running: ./rag_test/bot_test.py

## uvicorn app.main:app --reload
```
uvicorn app.main:app --reload
```

## LLM & RAG Configuration
    LLM: Mistral-7B-Instruct-v0.3

    Embeddings: sentence-transformers/all-MiniLM-L6-v2

    Frameworks: LangChain + FAISS

    Chain Type: RetrievalQA (stuff type)

The RAG engine is located in rag_engine.py and includes a custom prompt template that ensures grounded, context-based answers only.

## API Endpoints
** BASE URL
```
http://127.0.0.1:8000/api/
# POST /signup — Register a new user via Supabase
# POST /login — Log in and receive a JWT token
# Start a New Chat (POST /chat/start)
# Ask your qns (POST /chat/{chat_id})
# Chat History (/chat/{chat_id}/history)
# Chat List (/chats)
```



























