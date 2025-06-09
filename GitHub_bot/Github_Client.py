import streamlit as st
import requests

SERVER_URL = "http://localhost:8000"

st.title("GitHub Repo Analyzer")
st.markdown("Ask questions about a GitHub repository's code")


if 'repo_loaded' not in st.session_state:
    st.session_state.repo_loaded = False


repo_url = st.text_input(
    "GitHub Repository URL",
    value="https://github.com/campusx-official/langchain-document-loaders",
    help="Enter the full URL of the GitHub repository"
)

if st.button("Load Repository"):
    with st.spinner("Loading repository..."):
        try:
            response = requests.post(
                f"{SERVER_URL}/mcp/tool", 
                json={"repo_url": repo_url}
            )
            
            if response.status_code == 200:
                st.session_state.repo_loaded = True
                st.success("Repository loaded successfully!")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to server: {str(e)}")

if st.session_state.repo_loaded:
    question = st.text_input(
        "Ask a question about the repository",
        placeholder="What does this code do?"
    )
    
    if st.button("Ask"):
        if question.strip():
            with st.spinner("Finding answer..."):
                try:
                    response = requests.post(
                        f"{SERVER_URL}/mcp/ask", 
                        json={"question": question}
                    )
                    
                    if response.status_code == 200:
                        st.markdown("### Answer")
                        st.write(response.json()["answer"])
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to get response: {str(e)}")
        else:
            st.warning("Please enter a question")
else:
    st.info("Please load a GitHub repository first")