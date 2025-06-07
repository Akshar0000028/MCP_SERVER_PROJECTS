from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import requests
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()
class ResearchAgent:
    def __init__(self):
        self.llm = self.init_llm()
        self.tools = self.setup_tools()
        self.agent_executor = self.create_agent()

    def init_llm(self):
        return ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            nvidia_api_key=os.getenv["NVIDIA_API_KEY"],
            temperature=0.3
        )

    def setup_tools(self):
        @tool
        def search_arxiv_api(query: str, max_results: int = 5) -> str:
            """Search arXiv for research papers. Returns formatted results with URLs."""
            try:
                response = requests.post(
                    "http://localhost:8000/arxiv_query",
                    json={"query": query, "max_results": max_results}
                )
                response.raise_for_status()
                papers = response.json()
                
                if not papers:
                    return "No papers found matching your query."
                    
                formatted = []
                for paper in papers:
                    authors = ", ".join(paper['authors'][:3]) + ("." if len(paper['authors']) > 3 else "")
                    formatted.append(
                        f"Title: {paper['title']}\n"
                        f"Authors: {authors}\n"
                        f"Published: {paper.get('published', 'N/A')}\n"
                        f"URL: {paper.get('url', 'N/A')}\n"
                        f"Summary: {paper['summary'][:200]}...\n"
                    )
                return "\n\n".join(formatted)
            except Exception as e:
                return f"API Error: {str(e)}"

        @tool
        def summarize_arxiv_results(paper_results: str) -> str:
            """Generate a concise technical summary of research papers."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research scientist. Create a technical summary:
                - Identify key contributions
                - Note methodologies used
                - Highlight significant results
                - Keep under 300 words"""),
                ("user", "{papers}")
            ])
            chain = prompt | self.llm
            return chain.invoke({"papers": paper_results}).content

        return [search_arxiv_api, summarize_arxiv_results]

    def create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI research assistant with expertise in scientific papers.
            Your responses should:
            1. Be technically accurate
            2. Cite specific papers when available
            3. Include URLs to papers
            4. Maintain academic tone"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def query(self, input_text: str) -> str:
        """Public method to query the agent"""
        try:
            result = self.agent_executor.invoke({"input": input_text})
            return result["output"]
        except Exception as e:
            return f"Agent error: {str(e)}"
