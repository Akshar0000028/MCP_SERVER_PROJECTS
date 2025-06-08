import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import requests


def get_recommendations(query: str, top_k: int = 5) -> list:
    """Get product recommendations from the FastAPI server"""
    try:
        response = requests.post(
            "http://localhost:8000/mcp/tool/get_similar_products",
            json={"user_query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()["result"]
    except Exception as e:
        return [f"Error calling API: {str(e)}"]  


recommendation_tool = Tool(
    name="product_recommender",
    func=get_recommendations,
    description="Useful for finding similar products based on a user's query."
)


def setup_agent():
    llm = ChatNVIDIA(model="mixtral_8x7b",nvidia_api_key="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",)
    tools = [recommendation_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful e-commerce assistant. Provide clean, formatted product recommendations without any colored backgrounds."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    st.markdown("""
        <style>
            .stAlert { background-color: transparent !important; }
            .stException { background-color: transparent !important; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Product Recommender")
    st.write("Describe what you're looking for and we'll recommend similar products")
    
    agent = setup_agent()
    
    user_query = st.text_input("What products are you looking for?", 
                             placeholder="e.g. wireless headphones, running shoes...")
    
    if st.button("Find Recommendations"):
        if not user_query:
            st.warning("Please enter what you're looking for")
        else:
            with st.spinner("Finding similar products..."):
                result = agent.invoke({"input": f"Find 5 products similar to: {user_query}"})
                
                if isinstance(result["output"], list):
                    st.subheader("Top Recommendations")
                    for i, product in enumerate(result["output"], 1):
                        st.markdown(f"**{i}.** {product}")
                else:
                    
                    st.markdown(f"{result['output']}")

if __name__ == "__main__":
    main()