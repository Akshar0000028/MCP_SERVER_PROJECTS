from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel
from agent import ResearchAgent
import uvicorn
from typing import Optional
import requests

app = FastMCP(
    title="Research Agent MCP",
    description="arXiv Research Agent",
    version="1.0.0"
)

class AgentRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    detailed: Optional[bool] = False

agent = ResearchAgent()

@app.post("/query", tags=["Research"])
async def query_agent(request: AgentRequest):
    """
    Query the research agent with a scientific question
    
    """
    try:
        result = agent.query(f"{request.query} (return {request.max_results} papers)")
        
        if request.detailed:
            
            papers = requests.post(
                "http://localhost:8000/arxiv_query",
                json={"query": request.query, "max_results": request.max_results}
            ).json()
            return {
                "summary": result,
                "papers": papers
            }
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_config=None
    )