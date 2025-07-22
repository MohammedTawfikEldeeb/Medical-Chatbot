from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files , filter_to_minimal_documents , text_splitter , embedding_model
from pinecone import Pinecone
load_dotenv()



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN



extracted_data = load_pdf_files(data='data/')
filter_data = filter_to_minimal_documents(extracted_data)
text_chunks = text_splitter(filter_data)

embeddings = embedding_model()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(
    api_key= pinecone_api_key
)




index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)


stored_docs = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)