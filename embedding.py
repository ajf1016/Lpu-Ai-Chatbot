import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os

loader = TextLoader('./test.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()


# Initialize Pinecone client
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='us-east-1'
)


# # Define Index Name
# index_name = "langchain-demo"

# # Checking Index
# if index_name not in pinecone.list_indexes():
#     # Create new Index
#     pinecone.create_index(name=index_name, metric="cosine", dimension=768)
#     docsearch = Pinecone.from_documents(
#         docs, embeddings, index_name=index_name)
# else:
#     # Link to the existing index
#     docsearch = Pinecone.from_existing_index(index_name, embeddings)


api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError(
        "PINECONE_API_KEY is not set. Please check your environment variables.")
print(f"Using API Key: {api_key[:5]}*****")


os.environ["PINECONE_API_KEY"] = "your_actual_api_key"
