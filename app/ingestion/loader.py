import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader

def load_document(path):
    """Loads documents from a directory supporting PDF, DOCX, TXT, and PPTX."""
    documents = []
    if not os.path.exists(path):
        return []
        
    for file in os.listdir(path):
        fpath = os.path.join(path, file)
        if os.path.isdir(fpath):
            continue
            
        ext = file.lower().split('.')[-1]
        try:
            docs = []
            if ext == 'pdf':
                loader = PyPDFLoader(fpath)
                docs = loader.load()
            elif ext == 'docx':
                loader = Docx2txtLoader(fpath)
                docs = loader.load()
            elif ext == 'txt':
                loader = TextLoader(fpath, encoding='utf-8')
                docs = loader.load()
            elif ext == 'pptx':
                loader = UnstructuredPowerPointLoader(fpath)
                docs = loader.load()
                
            for doc in docs:
                doc.metadata['type'] = 'local'
                doc.metadata['title'] = file
                
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    return documents