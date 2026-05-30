import os
from embeddings import get_embedding_model
from ingestion import chunk_documents, load_all_pdfs
from vector_db import create_db, load_db
from llm import answer_query, get_model
from memory import rewrite_query
from router import route_query
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "./data"
CHUNK_SIZE = 550
CHUNK_OVERLAP = 150
DB_PATH = "./chroma_db"

# rag pipeline: load → chunk → embed → store → retrieve → generate


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("loading embedding model...")
    embedding_model = get_embedding_model()

    if not os.path.exists(DB_PATH):
        print("Creating vector DB...\n")

        # 1.LOAD
        documents, loaded_files = load_all_pdfs(DATA_DIR)
        if not documents:
            print("No PDFs found in data/ folder. Please add PDF files and try again.")
            return       
        print(f"\nLoaded PDFs: {loaded_files}")


        # 2. CHUNK
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"Number of chunks created: {len(chunks)}")

        # 3. EMBED AND 4. STORE
        print("creating vector store...")
        vector_store = create_db(chunks, embedding_model)
    else:
        print("Loading existing vector DB...\n")
        vector_store = load_db(embedding_model)

    available_pdfs = [os.path.basename(f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    print(f"Available PDFs: {available_pdfs}")
    chat_history = []  # to maintain conversation history
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # 5. RETRIEVE
        rewritten = rewrite_query(query, chat_history, get_model)
        if rewritten != query:
            print(f"\n[Rewritten query: {rewritten}]")
    
        selected_files = route_query(rewritten, available_pdfs)
        print(f"\n[Selected PDFs for retrieval: {selected_files}]")
    
        results = vector_store.similarity_search_with_score(
            rewritten, k=7,
            filter={"source": {"$in": [os.path.join(DATA_DIR, f) for f in selected_files]}} # type: ignore
        )  

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n--- Chunk {i} (score: {score:.4f}) ---")
            print(doc.page_content)

        # 6. GENERATE
        response = answer_query(results, query, chat_history)
        chat_history.append({"human": query, "assistant": response})

if __name__ == "__main__":
    main()