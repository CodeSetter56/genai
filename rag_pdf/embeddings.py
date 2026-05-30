import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pydantic import SecretStr

def get_embedding_model():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=SecretStr(os.environ["HUGGINGFACEHUB_API_TOKEN"]),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )