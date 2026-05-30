import time
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db"

def create_db(chunks, embedding_model):

    batch_size = 50
    vector_store = None

    for i in range(0, len(chunks), batch_size): # skips by batch size
        batch_chunks = chunks[i:i + batch_size] # gets the batch of chunks

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch_chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
        else:
            vector_store.add_documents(batch_chunks)

        # avoid rate limit
        if i + batch_size < len(chunks):
                    time.sleep(60)
                
    return vector_store


def load_db(embedding_model):

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    return vector_store

