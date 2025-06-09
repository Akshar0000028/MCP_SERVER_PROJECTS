import streamlit as st
import requests

st.title("Resume Analyzer")
st.write("Upload your resume PDF for analysis")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and st.button("Analyze"):
    with st.spinner("Analyzing..."):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                "http://localhost:8000/analyze_resume",
                files={"file": (uploaded_file.name, files["file"])}
            )
            analysis = response.json()["analysis"]
            st.write("Analysis Results:")
            st.write(analysis)
        except:
            st.error("Failed to analyze. Make sure the server is running.")