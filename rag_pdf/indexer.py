import os
import json
from ingestion import chunk_documents, load_all_pdfs

DESCRIPTIONS_FILE = "./chroma_db/pdf_descriptions.json"
INDEXED_FILES_FILE = "./chroma_db/indexed_files.json"

def load_descriptions():
    if os.path.exists(DESCRIPTIONS_FILE):
        with open(DESCRIPTIONS_FILE) as f:
            return json.load(f)
    return {}

def save_descriptions(pdf_descriptions):
    with open(DESCRIPTIONS_FILE, "w") as f:
        json.dump(pdf_descriptions, f)

def load_indexed_files():
    if os.path.exists(INDEXED_FILES_FILE):
        with open(INDEXED_FILES_FILE) as f:
            return json.load(f)
    # fallback: derive from descriptions file which already exists
    if os.path.exists(DESCRIPTIONS_FILE):
        with open(DESCRIPTIONS_FILE) as f:
            return list(json.load(f).keys())
    return []

def save_indexed_files(indexed_files):
    with open(INDEXED_FILES_FILE, "w") as f:
        json.dump(indexed_files, f)

def get_new_pdfs(data_dir, indexed_files):
    available_pdfs = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    return [f for f in available_pdfs if f not in indexed_files]

def get_removed_pdfs(data_dir, indexed_files):
    available_pdfs = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    return [f for f in indexed_files if f not in available_pdfs]

def load_new_pdf_chunks(data_dir, new_pdfs, chunk_size, chunk_overlap):
    # reuses load_all_pdfs with a filter instead of loading individually
    new_documents, _ = load_all_pdfs(data_dir, only_files=new_pdfs)
    return chunk_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def remove_pdfs_from_index(vector_store, data_dir, removed_pdfs, pdf_descriptions, indexed_files):
    for pdf_file in removed_pdfs:
        print(f"Removing {pdf_file} from index...")

        # delete chunks from chroma by source metadata filter
        vector_store.delete(where={"source": os.path.join(data_dir, pdf_file)})

        # remove from descriptions and indexed files
        pdf_descriptions.pop(pdf_file, None)
        indexed_files.remove(pdf_file)

    save_descriptions(pdf_descriptions)
    save_indexed_files(indexed_files)

    return pdf_descriptions, indexed_files