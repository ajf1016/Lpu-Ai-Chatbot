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
            model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=500,
            n_ctx=2048,
            n_batch=128,
            n_threads=4,
            verbose=True  # Set to True to debug model output
        )
        
        # RAG Chain with simplified parameters
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,  # Reduced for more focused results
                    "score_threshold": 0.2  # Lower threshold to catch more matches
                }
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Predefined responses for common queries
        self.greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        self.farewells = ["bye", "goodbye", "see you", "exit", "quit"]
        
        # Common admission questions with predefined answers
        self.faq = {
            "btech": "To apply for B.Tech at LPU, you need to: 1) Complete the online application form at LPU's website, 2) Pay the application fee, 3) Take the LPU entrance test or submit scores from JEE/other recognized exams, 4) Wait for admission offer based on your performance. For more details, please visit the LPU website or contact the admissions office.",
            "admission": "LPU offers admissions through both entrance exams and direct applications. You can apply online through the LPU website. Required documents typically include academic transcripts, entrance exam scores, ID proof, and photographs. Specific requirements vary by program.",
            "fees": "LPU's fee structure varies by program. B.Tech programs typically range from 1.5-2.5 lakhs per year depending on specialization. Exact fee details are available on the official LPU website or by contacting the admissions office.",
            "scholarship": "LPU offers various scholarships based on academic merit, sports achievements, and financial need. Scholarships can range from 10% to 100% of tuition fees depending on eligibility criteria. You can apply for scholarships during the admission process.",
            "hostel": "LPU provides on-campus hostel facilities with various accommodation options. Rooms include single, double, and triple sharing with amenities like Wi-Fi, laundry services, and dining facilities. Hostel fees vary based on the type of accommodation chosen."
        }
        
    def ask(self, question):
        # Handle greetings
        if question.lower().strip() in self.greetings:
            return "Hello! I'm the LPU Admissions Assistant. How can I help you with admissions information today?"
        
        # Handle farewells
        if question.lower().strip() in self.farewells:
            return "Goodbye! Feel free to return if you have more questions about LPU admissions."
        
        # Check for FAQ matches first
        for keyword, answer in self.faq.items():
            if keyword in question.lower():
                return answer
            
        try:
            # Use RAG for all other queries
            result = self.rag_chain.invoke(question)
            answer = result.get("result", "")
            sources = result.get("source_documents", [])
            
            # Check answer quality
            if not answer or len(answer.strip()) < 20 or "I don't know" in answer:
                # Fall back to a generic helpful response
                return "I don't have specific information about that in my knowledge base. For detailed information about LPU admissions, I recommend visiting the official LPU website or contacting the admissions office directly at admissions@lpu.co.in."
                
            return answer
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I apologize for the inconvenience. I encountered an error processing your question. Please try asking a different question about LPU admissions."

def main():
    bot = LPUAdmissionsChatbot('./lpu_admissions.txt')
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