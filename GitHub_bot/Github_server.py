from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from embedder import create_retriever
import os, shutil, stat
from git import Repo, GitCommandError

class RepoRequest(BaseModel):
    repo_url: str

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()
mcp = FastMCP(app)


qa_chains = {}

def force_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

@app.post("/mcp/tool")
async def load_repo(payload: RepoRequest):
    try:
        repo_name = payload.repo_url.rstrip("/").split("/")[-1]
        clone_path = f"./repos/{repo_name}"

        os.makedirs("./repos", exist_ok=True)

        if os.path.exists(clone_path):
            shutil.rmtree(clone_path, onerror=force_remove_readonly)

        print(f"Cloning repo from {payload.repo_url} to {clone_path}")
        Repo.clone_from(payload.repo_url, clone_path)

        retriever = create_retriever(clone_path)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatNVIDIA(
                model="meta/llama3-70b-instruct",
                nvidia_api_key="NVIDIA_API_KEY" 
            ),
            retriever=retriever,
            return_source_documents=True
        )

        
        qa_chains[repo_name] = qa_chain
        qa_chains["active"] = qa_chain  

        return {"message": f" Repo loaded successfully: {repo_name}"}

    except GitCommandError as e:
        raise HTTPException(status_code=400, detail=f"Git error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/ask")
async def ask_question(payload: QuestionRequest):
    if not qa_chains:
        raise HTTPException(status_code=400, detail="No repository loaded.")
    
    qa_chain = qa_chains.get("active")
    
    if not qa_chain:
        raise HTTPException(status_code=400, detail="No active repository loaded. Please load a repository first.")

    try:
        result = qa_chain.invoke({"query": payload.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Github_server:app", host="127.0.0.1", port=8000, reload=False) 
