import os
from embeddings import get_embedding_model
from ingestion import chunk_documents, load_pdf
from vector_db import create_db, load_db
from llm import answer_query
from dotenv import load_dotenv

load_dotenv()

PDF_FILE = "./data/tcs ol.pdf"
CHUNK_SIZE = 550
CHUNK_OVERLAP = 150
DB_PATH = "./chroma_db"

# rag pipeline: load → chunk → embed → store → retrieve → generate


def main():
    os.makedirs("./data", exist_ok=True)    

    print("loading embedding model...")
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_PATH):
        if not os.path.exists(PDF_FILE):
            print(f"No PDF found at {PDF_FILE}.")
            return
        print("Creating vector DB...\n")

        # 1.LOAD
        print("loading pdf...")
        documents = load_pdf(PDF_FILE)
        print("chunking documents...")

        # 2. CHUNK
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Number of chunks created: {len(chunks)}")

        # 3. EMBED AND 4. STORE
        print("creating vector store...")
        vector_store = create_db(chunks, embedding_model)
    else:
        print("Loading existing vector DB...\n")
        vector_store = load_db(embedding_model)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # 5. RETRIEVE
        results = vector_store.similarity_search_with_score(query, k=5)  # type: ignore

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n--- Chunk {i} (score: {score:.4f}) ---")
            print(doc.page_content)

        # 6. GENERATE
        answer_query(results, query)

if __name__ == "__main__":
    main()