import os
from dotenv import load_dotenv
from langchain_community.llms import LlamaCpp

class LPUAdmissionsChatbot:
    def __init__(self, model_path="./models/phi-2.Q4_K_M.gguf"):
        load_dotenv()
        
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
        
        Question: {question}
        
        Answer:
        """
        
        # Basic responses
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
            # Format prompt with the question
            prompt = self.system_prompt.format(question=question)
            
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
    chatbot = LPUAdmissionsChatbot('./models/phi-2.Q4_K_M.gguf')
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