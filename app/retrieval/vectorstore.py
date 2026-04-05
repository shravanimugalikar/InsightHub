from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    return db
