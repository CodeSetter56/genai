import glob # glob is a standard Python library that finds files matching a pattern, like a search with wildcards
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    return documents

def load_all_pdfs(data_dir):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        return [], []
    
    all_documents = []
    loaded_files = []
    
    for pdf_file in pdf_files:
        print(f"loading {os.path.basename(pdf_file)}...")
        docs = load_pdf(pdf_file)
        all_documents.extend(docs)
        loaded_files.append(os.path.basename(pdf_file))
    
    return all_documents, loaded_files

def chunk_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks