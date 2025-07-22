from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings


## Extract data from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )

    documents = loader.load()
    return documents


## Filter extracted data to minimal documents
def filter_to_minimal_documents(docs : List[Document]) -> List[Document]:
    minimal_docs : List[Document] = []

    for doc in docs :
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata={"source": src}
            )
        )
    
    return minimal_docs



## Split extracted data into chunks
def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


## Download Embedding model
def embedding_model():

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )

    return embeddings
