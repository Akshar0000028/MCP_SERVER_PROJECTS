from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    nvidia_api_key=""
)

db = FAISS.load_local(
    "D:\\MCP SERVER PROJECTS\\E-commerce_Product_Recommender\\faiss_index",embeddings,allow_dangerous_deserialization=True)

app = FastAPI()
mcp = FastMCP(app=app)

class RecommendRequest(BaseModel):
    user_query: str
    top_k: int = 5

@app.post("/health")
def health_check():
    return {"status": "ok"}


@mcp.tool()
def get_similar_products(user_query: str, top_k: int = 5):
    """Retrieve top-K similar products to the user query."""
    docs = db.similarity_search(user_query, k=top_k)
    return [doc.page_content for doc in docs]


app.mount("/mcp", mcp.http_app())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
