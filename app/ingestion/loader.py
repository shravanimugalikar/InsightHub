import os
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

def load_document(path):
    loader = PyPDFDirectoryLoader(path)

    return loader.load()
    