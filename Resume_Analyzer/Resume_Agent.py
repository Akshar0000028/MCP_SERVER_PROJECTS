import requests
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_nvidia_ai_endpoints import ChatNVIDIA

MCP_SERVER_URL = "http://localhost:8000/analyze_resume"

@tool
def analyze_resume_tool(file_path: str) -> str:
    """Uploads a resume PDF to the server and returns analysis."""
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(MCP_SERVER_URL, files=files)
            response.raise_for_status()
            return response.json()['analysis']
    except Exception as e:
        return f"Error analyzing resume: {str(e)}"

llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    nvidia_api_key="NVIDIA_API_KEY",
    temperature=0.3
)

tools = [analyze_resume_tool]

agent = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    file_path = "C:\\Users\\Akshar Savaliya\\Downloads\\ADIT_AI_SAVALIYAAKSHARKUMARKIRANBHAI.pdf"
    try:
        result = agent.invoke({"input": f"Please analyze this resume: {file_path}"})
        print(result['output'])
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
