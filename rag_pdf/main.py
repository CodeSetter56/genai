import os
from embeddings import get_embedding_model
from ingestion import chunk_documents, load_all_pdfs
from indexer import (load_descriptions, save_descriptions,
                     load_indexed_files, save_indexed_files,
                     get_new_pdfs, get_removed_pdfs,
                     load_new_pdf_chunks, remove_pdfs_from_index)
from vector_db import create_db, load_db, add_to_db
from llm import answer_query, get_model
from memory import rewrite_query
from router import generate_pdf_description, route_query
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "./data"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 300
DB_PATH = "./chroma_db"

# rag pipeline: load → chunk → embed → store → retrieve → generate

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("loading embedding model...")
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_PATH):
        print("Creating vector DB...\n")

        # 1. LOAD: load all PDFs from data/ and extract text
        documents, loaded_files = load_all_pdfs(DATA_DIR)
        if not documents:
            print("No PDFs found in data/ folder.")
            return
        print(f"\nLoaded PDFs: {loaded_files}")

        # 2. CHUNK: split documents into overlapping chunks for better retrieval performance
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Number of chunks created: {len(chunks)}")

        # 2.5. DESCRIBE: generate and save structured metadata per PDF for routing
        os.makedirs(DB_PATH, exist_ok=True)
        pdf_descriptions = {}
        for pdf_file in loaded_files:
            pdf_chunks = [c for c in chunks if os.path.basename(c.metadata["source"]) == pdf_file]
            print(f"Generating description for {pdf_file}...")
            pdf_descriptions[pdf_file] = generate_pdf_description(pdf_file, pdf_chunks)
        save_descriptions(pdf_descriptions)
        save_indexed_files(loaded_files)

        print("\nGenerated descriptions:")
        for name, desc in pdf_descriptions.items():
            print(f"  {name}: {desc}")

        # 3. EMBED AND 4. STORE
        print("Creating vector store...")
        vector_store = create_db(chunks, embedding_model)

    # 4.1. UPDATE DB: if DB already exists
    else:
        print("Loading existing vector DB...\n")

        vector_store = load_db(embedding_model)
        pdf_descriptions = load_descriptions()
        indexed_files = load_indexed_files()

        # remove chunks and metadata for deleted PDFs
        removed_pdfs = get_removed_pdfs(DATA_DIR, indexed_files)
        if removed_pdfs:
            print(f"Removed PDFs detected: {removed_pdfs}")
            pdf_descriptions, indexed_files = remove_pdfs_from_index(
                vector_store, DATA_DIR, removed_pdfs, pdf_descriptions, indexed_files
            )

        # load, chunk, describe and index new PDFs
        new_pdfs = get_new_pdfs(DATA_DIR, indexed_files)
        if new_pdfs:
            print(f"New PDFs detected: {new_pdfs}\n")
            new_chunks = load_new_pdf_chunks(DATA_DIR, new_pdfs, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"Number of new chunks created: {len(new_chunks)}")

            for pdf_file in new_pdfs:
                pdf_chunks = [c for c in new_chunks if os.path.basename(c.metadata["source"]) == pdf_file]
                print(f"Generating description for {pdf_file}...")
                pdf_descriptions[pdf_file] = generate_pdf_description(pdf_file, pdf_chunks)

            # save before embedding in case embedding step fails
            save_descriptions(pdf_descriptions)
            save_indexed_files(indexed_files + new_pdfs)

            print("Adding new chunks to vector store...")
            vector_store = add_to_db(new_chunks, embedding_model)

        if not removed_pdfs and not new_pdfs:
            print("No changes detected. Using existing vector DB.")

    # sanity check to ensure vector store is available before entering query loop
    assert vector_store is not None, "vector_store failed to initialize."
    print(f"\nAvailable PDFs: {list(pdf_descriptions.keys())}")

    chat_history = []

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # 4.2. REWRITE: rephrase follow-up queries into standalone questions
        rewritten = rewrite_query(query, chat_history, get_model)
        if rewritten != query:
            print(f"\n[Rewritten query: {rewritten}]")

        # 4.3. ROUTE: select relevant PDFs for this query
        selected_files = route_query(rewritten, pdf_descriptions)
        print(f"\n[Selected PDFs for retrieval: {selected_files}]")

        # 5. RETRIEVE: similarity search filtered to selected PDFs
        results = vector_store.similarity_search_with_score(
            rewritten, k=7,
            filter={"source": {"$in": [os.path.join(DATA_DIR, f) for f in selected_files]}}  # type: ignore
        )

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n--- Chunk {i} (score: {score:.4f}) ---")
            print(doc.page_content)

        # 6. GENERATE: answer using retrieved chunks and chat history
        response = answer_query(results, query, chat_history)
        chat_history.append({"human": query, "assistant": response})

if __name__ == "__main__":
    main()