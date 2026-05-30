import os
from embeddings import get_embedding_model
from ingestion import chunk_documents, load_pdf
from vector_db import create_db, load_db
from dotenv import load_dotenv

load_dotenv()

PDF_FILE = "./data/tcs ol.pdf"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 75

DB_PATH = "./chroma_db"


def main():

    # 1 load embedding model
    print("loading embedding model...")
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_PATH):
        print("Creating vector DB...\n")
        
        # 2 load and chunk documents
        print("loading pdf...")
        documents = load_pdf(PDF_FILE)

        print("chunking documents...")
        chunks = chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # 3 create or load vector store
        print(f"Number of chunks created: {len(chunks)}")
        print("creating vector store...")
        vector_store = create_db(
            chunks,
            embedding_model
        )
    else:
        print("Loading existing vector DB...\n")
        vector_store = load_db(
            embedding_model
        )

    while True:

        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # 4 retrieve relevant chunks using similarity search
        results = vector_store.similarity_search_with_score( # type: ignore
            query,
            k=3
        )

        print("\nTop Retrieved Chunks:\n")

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n--- Chunk {i} (score: {score:.4f}) ---")

            print("\nMETADATA:")
            print(doc.metadata)

            print("\nCONTENT:")
            print(doc.page_content)

            print("\n-----------------\n")


if __name__ == "__main__":
    main()