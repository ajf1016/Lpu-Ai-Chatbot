import os
import numpy as np
import torch
import faiss
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp

class LPUAdmissionsChatbot:
    def __init__(self, document_path, model_path="./models/phi-2.Q4_K_M.gguf"):
        load_dotenv()
        torch.set_num_threads(1)
        
        # Load text data
        self.text_chunks, self.metadata = self.process_documents(document_path)
        
        # Set up embeddings model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
        self.embedding_model = AutoModel.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
        
        # Create FAISS index
        self.index, self.embeddings = self.create_faiss_index()
        
        # LLM setup
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.3,
            max_tokens=500,
            n_ctx=2048,
            n_batch=128,
            n_threads=4,
            verbose=False
        )
        
        # System prompt
        self.system_prompt = """
        You are a helpful and friendly assistant for Lovely Professional University (LPU) admissions.
        Your job is to answer questions about LPU's admission processes, programs, fees, scholarships, 
        campus facilities, and other related information.
        
        Use only the context information to answer the question. If the context doesn't contain the answer,
        politely state that you don't have that specific information.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        # FAQ for common queries
        # self.faq = {
        #     # "btech": "To apply for B.Tech at LPU, you need to: 1) Complete the online application form at LPU's website, 2) Pay the application fee, 3) Take the LPU entrance test or submit scores from JEE/other recognized exams, 4) Wait for admission offer based on your performance.",
        #     "admission": "LPU offers admissions through both entrance exams and direct applications. Required documents typically include academic transcripts, entrance exam scores, ID proof, and photographs.",
        #     "fees": "LPU's fee structure varies by program. B.Tech programs typically range from 1.5-2.5 lakhs per year depending on specialization.",
        #     "scholarship": "LPU offers various scholarships based on academic merit, sports achievements, and financial need. Scholarships can range from 10% to 100% of tuition fees depending on eligibility criteria.",
        #     "hostel": "LPU provides on-campus hostel facilities with various accommodation options. Rooms include single, double, and triple sharing with amenities like Wi-Fi, laundry services, and dining facilities."
        # }
        
        # Basic responses
        self.greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        self.farewells = ["bye", "goodbye", "see you", "exit", "quit"]
    
    def process_documents(self, document_path):
        """Process text document into chunks for embedding"""
        loader = TextLoader(document_path)
        documents = loader.load()
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        doc_chunks = text_splitter.split_documents(documents)
        
        # Extract text and metadata
        text_chunks = [doc.page_content for doc in doc_chunks]
        metadata = [doc.metadata for doc in doc_chunks]
        
        return text_chunks, metadata
    
    def get_embedding(self, text):
        """Generate proper embedding for a piece of text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Generate actual embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use mean pooling of the last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.squeeze().detach().cpu().numpy()
    
    def create_faiss_index(self):
        """Create FAISS index from text chunks"""
        embeddings = []
        for chunk in self.text_chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype="float32")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Get the actual dimension from the embeddings
        dimension = embeddings_array.shape[1]
        
        # Create and populate FAISS index with inner product (cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        return index, embeddings_array
    
    def retrieve_relevant_chunks(self, query, k=3, threshold=0.7):
        """Retrieve the most relevant chunks for a query"""
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding], dtype="float32")
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        similarities, indices = self.index.search(query_embedding, k)
        
        # For IP/cosine, higher is better (opposite of L2 distance)
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            # For cosine similarity (IP with normalized vectors), higher is better
            if similarities[0][i] > threshold:
                relevant_chunks.append(self.text_chunks[idx])
        
        # If no chunks pass the threshold, return a few anyway
        if not relevant_chunks and len(indices[0]) > 0:
            relevant_chunks = [self.text_chunks[idx] for idx in indices[0][:2]]
            
        # Add debug info
        print(f"Retrieved {len(relevant_chunks)} chunks for query: '{query}'")
        for i, chunk in enumerate(relevant_chunks):
            print(f"Chunk {i} similarity: {similarities[0][i]:.4f}")
            print(f"Chunk {i} preview: {chunk[:100]}...")
        
        return relevant_chunks
    
    def ask(self, question):
        # Handle greetings
        if question.lower().strip() in self.greetings:
            return "Hello! I'm the LPU Admissions Assistant. How can I help you with admissions information today?"
        
        # Handle farewells
        if question.lower().strip() in self.farewells:
            return "Goodbye! Feel free to return if you have more questions about LPU admissions."
        
        # For mathematical questions or simple questions
        if any(op in question for op in ['+', '-', '*', '/', '=']) or len(question.split()) <= 3:
            if "+" in question and len(question.split()) <= 3:
                # Simple addition
                try:
                    nums = question.replace("?", "").strip().split("+")
                    result = sum(float(n) for n in nums)
                    return f"The answer is {result}"
                except:
                    pass  # If parsing fails, continue to regular processing
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(question)
            
            # If no relevant chunks found
            if not relevant_chunks:
                return ("I don't have specific information about that in my knowledge base. "
                        "For detailed information about LPU admissions, I recommend visiting "
                        "the official LPU website or contacting the admissions office.")
                
            context = "\n\n".join(relevant_chunks)
            
            # Format prompt with context
            prompt = self.system_prompt.format(context=context, question=question)
            
            # Print prompt for debugging
            print("\nPrompt sent to LLM:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
            
            # Get response from LLM
            response = self.llm.invoke(prompt).strip()
            
            # If response is too short or indicates no information
            if len(response) < 20 or "I don't have" in response or "no specific information" in response:
                return ("I don't have specific information about that in my knowledge base. "
                        "For detailed information about LPU admissions, I recommend visiting "
                        "the official LPU website or contacting the admissions office.")
            
            return response
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I apologize for the inconvenience. I encountered an error processing your question."

def main():
    chatbot = LPUAdmissionsChatbot('./new_context.txt')
    print("LPU Admissions Assistant ready! (Type 'exit' to quit)")
    
    while True:
        query = input("\nAsk me about LPU admissions: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using the LPU Admissions Assistant. Goodbye!")
            break
        
        answer = chatbot.ask(query)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()