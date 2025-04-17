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
            chunk_size=800,  # Smaller chunks for better retrieval
            chunk_overlap=200  # More overlap to maintain context
        )
        docs = text_splitter.split_documents(documents)
        
        # Embeddings & Vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Simplified prompt to help the small model focus
        prompt = PromptTemplate(
            template="""
            You are an admissions assistant for Lovely Professional University (LPU).
            
            ###Context:
            {context}
            
            ###Question:
            {question}
            
            ###Answer:
            """,
            input_variables=["context", "question"]
        )
        
        # LLM setup
        llm = LlamaCpp(
            model_path="./models/phi-2.Q4_K_M.gguf",
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=500,
            n_ctx=2048,
            n_batch=128,
            n_threads=4,
            verbose=True 
        )
        
        # RAG Chain with simplified parameters
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,  # Increase number of retrieved documents for better accuracy
                    "score_threshold": 0.3  # Adjust score threshold to be more inclusive
                }
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Predefined responses for common queries
        self.greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        self.farewells = ["bye", "goodbye", "see you", "exit", "quit"]
        
    def ask(self, question):
        # Handle greetings
        if question.lower().strip() in self.greetings:
            return "Hello! I'm the LPU Admissions Assistant. How can I help you with admissions information today?"
        
        # Handle farewells
        if question.lower().strip() in self.farewells:
            return "Goodbye! Feel free to return if you have more questions about LPU admissions."
        
        try:
            # Use RAG for all other queries
            result = self.rag_chain.invoke(question)
            answer = result.get("result", "")
            sources = result.get("source_documents", [])
            
            # Check if the answer is empty or too generic
            if not answer or len(answer.strip()) < 20 or "I don't know" in answer:
                # Fall back to a generic helpful response
                return "I don't have specific information about that in my knowledge base. For detailed information about LPU admissions, I recommend visiting the official LPU website or contacting the admissions office directly at admissions@lpu.co.in."
            
            # Return the answer with the relevant context if available
            return f"{answer}\n\nSources: {', '.join([source['metadata']['source'] for source in sources])}"
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I apologize for the inconvenience. I encountered an error processing your question. Please try asking a different question about LPU admissions."

def main():
    bot = LPUAdmissionsChatbot('./new_context.txt')
    print("LPU Admissions Assistant ready! (Type 'exit' to quit)")
    
    while True:
        query = input("\nAsk me about LPU admissions: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the LPU Admissions Assistant. Goodbye!")
            break
        
        answer = bot.ask(query)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
    