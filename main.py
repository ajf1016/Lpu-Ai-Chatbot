import os
from dotenv import load_dotenv
import torch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate


class LPUAdmissionsChatbot:
    def __init__(self, document_path):
        load_dotenv()
        torch.set_num_threads(1)

        # Load and split documents
        loader = TextLoader(document_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for more precise retrieval
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)

        # Embeddings & Vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Strict Prompt Template
        strict_prompt = PromptTemplate(
            template="""
You are a precise admissions assistant for Lovely Professional University (LPU).
IMPORTANT RULES:
1. ONLY use the provided context to answer questions.
2. DO NOT generate information not in the context.
3. If the context does not contain a clear answer, respond with: "Sorry, I don't have specific information about this."

Context: {context}

Question: {question}

Answer STRICTLY based on the context:""",
            input_variables=["context", "question"]
        )

        # LLM with lower temperature for more conservative responses
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={
                "temperature": 0.1,  # Lowered temperature to reduce creativity
                "max_new_tokens": 300  # Reduced token count
            },
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # RAG Chain with stricter retrieval
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 2,  # Limit to top 2 most relevant documents
                    "score_threshold": 0.7  # Only retrieve highly relevant documents
                }
            ),
            chain_type_kwargs={"prompt": strict_prompt},
            return_source_documents=True
        )

    def ask(self, question):
        # Preprocess the question to ensure it's about LPU admissions
        # processed_question = f"LPU admissions: {question}"
        processed_question = f"{question}"

        try:
            result = self.rag_chain.invoke(processed_question)
            answer = result.get(
                "result", "Sorry, I don't have specific information about this.")
            sources = result.get("source_documents", [])

            # Additional validation
            if not sources or all(len(doc.page_content.strip()) < 50 for doc in sources):
                return "Sorry, I could not find relevant information about this LPU admission query."

            return answer

        except Exception as e:
            print(f"Error processing query: {e}")
            return "Sorry, I encountered an error processing your question."


def main():
    # Use the document you provided
    bot = LPUAdmissionsChatbot('./lpu_admissions.txt')

    while True:
        query = input("Ask me about LPU admissions: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break

        answer = bot.ask(query)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
