from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel

recipes_df = pd.read_csv("D:\\MCP SERVER PROJECTS\\Receipe_Recommendation\\recipe_metadata.csv")
index = faiss.read_index("D:\\MCP SERVER PROJECTS\\Receipe_Recommendation\\recipe_faiss.index")
model = SentenceTransformer('all-MiniLM-L6-v2')


app = FastAPI()
mcp = FastMCP(app=app)


class RecipeRequest(BaseModel):
    ingredients: str
    top_k: int = 5

def recommend_recipes(ingredients: str, top_k: int = 5) -> List[str]:
    processed_input = ingredients.lower()
    query_embedding = model.encode([processed_input])
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, top_k)
    return [recipes_df.iloc[idx]['recipe_name'] for idx in indices[0]]


@app.post("/recommend")
def recommend_endpoint(request: RecipeRequest):
    try:
        recipes = recommend_recipes(request.ingredients, request.top_k)
        return recipes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool()
def RecipeRecommender(ingredients: str, top_k: int = 5) -> List[str]:
    return recommend_recipes(ingredients, top_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
