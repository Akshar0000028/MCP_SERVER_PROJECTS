import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import requests

def execute_sql_query(query: str) -> str:
    """Executes a SQL query on a remote FastAPI server and returns the results."""
    try:
        response = requests.post("http://localhost:8000/query", json={"query": query})
        data = response.json()
        if "error" in data:
            return f"Error: {data['error']}"
        if not data["rows"]:
            return "Query successful, but no data found."
        return f"Columns: {data['columns']}\n\n Rows: {data['rows']}"
    except Exception as e:
        return f"Client error: {str(e)}"

tool = Tool.from_function(
    func=execute_sql_query,
    name="execute_sql_query",
    description="Executes a SQL query on a remote FastAPI server and returns the results."
)


llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    nvidia_api_key="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",  
    temperature=0.3
)

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)


st.title("SQL Query Assistant")

user_input = st.text_input("Ask your question (e.g., 'List all employees')", key="input")

if st.button("Run Query"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = agent.run(user_input)
                st.success("Result:")
                st.text(result)
            except Exception as e:
                st.error(f"Agent error: {str(e)}")
