import os
from dotenv import load_dotenv
import torch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate


class LPUAdmissionsChatbot:
    def __init__(self, document_path):
        load_dotenv()
        torch.set_num_threads(1)
        # Load and split documents
        loader = TextLoader(document_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=100  # Increased overlap
        )
        docs = text_splitter.split_documents(documents)
        # Embeddings & Vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Updated Prompt Template - less restrictive
        prompt = PromptTemplate(
            template="""
            You are a helpful admissions assistant for Lovely Professional University (LPU).
            
            Context information from LPU documents:
            {context}
            
            Question: {question}
            
            Please provide a helpful answer based primarily on the context information.
            If the context doesn't contain sufficient information, acknowledge this
            and provide a brief general response that might be helpful.
            """,
            input_variables=["context", "question"]
        )

        # LLM - local TinyLlama
        llm = LlamaCpp(
            model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            temperature=0.3,
            max_tokens=300,
            n_ctx=2048,
            n_batch=128,
            n_threads=4,  # Tune this as per your MacBook
            verbose=False
        )

        # Less restrictive RAG Chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,  # Increased number of documents
                    "score_threshold": 0.4  # Lower threshold for more matches
                }
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def ask(self, question):
        try:
            result = self.rag_chain.invoke(question)
            answer = result.get(
                "result", "I don't have specific information about this in my knowledge base.")
            sources = result.get("source_documents", [])

            # Less restrictive validation
            if not sources:
                return "I couldn't find relevant information about this query in the LPU documents."

            return answer
        except Exception as e:
            print(f"Error processing query: {e}")
            return "Sorry, I encountered an error processing your question."


def main():
    bot = LPUAdmissionsChatbot('./lpu_admissions.txt')
    while True:
        query = input("Ask me about LPU admissions: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        answer = bot.ask(query)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
