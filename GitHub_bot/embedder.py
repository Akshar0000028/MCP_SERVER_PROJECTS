from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os

def load_codebase(path: str):
    loader = DirectoryLoader(path, glob="**/*.py", loader_cls=TextLoader)
    return loader.load()

def create_retriever(repo_path: str):
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Path {repo_path} does not exist.")

    docs = load_codebase(repo_path)
    if not docs:
        raise ValueError(f"No Python files found in {repo_path}.")

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        model_type="passage",
        nvidia_api_key="NVIDIA_API_KEY"  
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever()
