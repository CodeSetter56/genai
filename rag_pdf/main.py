import os
import json
from embeddings import get_embedding_model
from ingestion import chunk_documents, load_all_pdfs
from vector_db import create_db, load_db
from llm import answer_query, get_model
from memory import rewrite_query
from router import generate_pdf_description, route_query
from rerank import rerank_documents # Import the new rerank function
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "./data"
CHUNK_SIZE = 550
CHUNK_OVERLAP = 150
DB_PATH = "./chroma_db"
DESCRIPTIONS_FILE = "./chroma_db/pdf_descriptions.json"

# rag pipeline: load → chunk → embed → store → retrieve → generate

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("loading embedding model...")
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_PATH):
        print("Creating vector DB...\n")

        # 1. LOAD
        documents, loaded_files = load_all_pdfs(DATA_DIR)
        if not documents:
            print("No PDFs found in data/ folder.")
            return

        print(f"\nLoaded PDFs: {loaded_files}")

        # 2. CHUNK
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Number of chunks created: {len(chunks)}")

        # 2.5. GENERATE AND SAVE DESCRIPTIONS
        os.makedirs(DB_PATH, exist_ok=True)
        pdf_descriptions = {}
        for pdf_file in loaded_files:
            pdf_chunks = [c for c in chunks if os.path.basename(c.metadata["source"]) == pdf_file]
            print(f"Generating description for {pdf_file}...")
            pdf_descriptions[pdf_file] = generate_pdf_description(pdf_file, pdf_chunks)

        with open(DESCRIPTIONS_FILE, "w") as f:
            json.dump(pdf_descriptions, f)

        print("\nGenerated descriptions:")
        for name, desc in pdf_descriptions.items():
            print(f"  {name}: {desc}")

        # 3. EMBED AND 4. STORE
        print("creating vector store...")
        vector_store = create_db(chunks, embedding_model)

    else:
        print("Loading existing vector DB...\n")
        vector_store = load_db(embedding_model)

        # load descriptions of pdfs or use filenames as fallback
        if os.path.exists(DESCRIPTIONS_FILE):
            with open(DESCRIPTIONS_FILE) as f:
                pdf_descriptions = json.load(f)
        else:
            available_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
            pdf_descriptions = {pdf: pdf for pdf in available_pdfs}

    assert vector_store is not None, "vector_store failed to initialize."
    print(f"Available PDFs: {list(pdf_descriptions.keys())}")

    chat_history = []

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # 4.5. REWRITE QUERY
        rewritten = rewrite_query(query, chat_history, get_model)
        if rewritten != query:
            print(f"\n[Rewritten query: {rewritten}]")

        # 5. ROUTE AND RETRIEVE
        selected_files = route_query(rewritten, pdf_descriptions)
        print(f"\n[Selected PDFs for retrieval: {selected_files}]")

        results = vector_store.similarity_search_with_score(
            rewritten, k=7,
            # filter by selected PDF files
            filter={"source": {"$in": [os.path.join(DATA_DIR, f) for f in selected_files]}}  # type: ignore
        )

        # RERANKING STEP: Re-evaluate and reorder the retrieved documents
        if results:
            print(f"\nReranking {len(results)} retrieved chunks...")
            # We pass the rewritten query to the reranker for better context
            # and request the top 5 most relevant chunks after reranking.
            reranked_results = rerank_documents(rewritten, results, top_n=5)
        else:
            reranked_results = [] # No results to rerank

        # Print the reranked chunks
        for i, (doc, score) in enumerate(reranked_results, start=1):
            print(f"\n--- Chunk {i} (reranked score: {score:.4f}) ---")
            print(doc.page_content)

        # 6. GENERATE
        # Pass the reranked results to the answer_query function
        response = answer_query(reranked_results, query, chat_history)
        chat_history.append({"human": query, "assistant": response})

if __name__ == "__main__":
    main()
