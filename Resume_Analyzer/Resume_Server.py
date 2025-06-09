from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile
import uvicorn

app = FastAPI()

@app.post("/analyze_resume")
async def analyze_resume(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])

        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume reviewer. Provide detailed feedback and improvement suggestions."),
            ("human", "{resume_text}")
        ])

        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            nvidia_api_key="NVIDIA_API_KEY",
            temperature=0.3
        )
        
        chain = prompt | llm
        result = chain.invoke({"resume_text": text})

        return {"analysis": result.content}
    finally:
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
