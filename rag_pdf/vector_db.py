from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "./chroma_db"

def create_db(chunks, embedding_model):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


def load_db(embedding_model):

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    return vector_store