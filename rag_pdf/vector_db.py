import time
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db"

def create_db(chunks, embedding_model):
    batch_size = 50
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        # batching chunks to avoid rate limits and memory issues when embedding large documents
        batch_chunks = chunks[i:i + batch_size]

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch_chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
        else:
            vector_store.add_documents(batch_chunks)

        # avoiding rate limit
        if i + batch_size < len(chunks):
            time.sleep(60)

    if vector_store is None: # ensure vector_store is created even if chunks is empty
        raise ValueError("No chunks provided to create_db.")

    return vector_store


def load_db(embedding_model):

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    return vector_store

def add_to_db(chunks, embedding_model):
    vector_store = load_db(embedding_model)
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        vector_store.add_documents(batch_chunks)

        if i + batch_size < len(chunks):
            time.sleep(60)  # avoid rate limit

    return vector_store