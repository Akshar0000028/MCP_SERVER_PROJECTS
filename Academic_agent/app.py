import streamlit as st
import arxiv

def main():
    st.title("arXiv Paper Search")
    
   
    search_query = st.text_input("Enter your search query:")
    
    if st.button("Search"):
        if search_query:
        
            search = arxiv.Search(
                query=search_query,
                max_results=10  
            )
    
            for result in search.results():
                st.write("---")
                st.write(f"*Title:* {result.title}")
                st.write(f"*Authors:* {', '.join(author.name for author in result.authors)}")
                st.write(f"*Published:* {result.published.strftime('%d %B %Y')}")
                st.write(f"*Summary:* {result.summary[:200]}...")
                st.write(f"*PDF Link:* {result.pdf_url}")
        else:
            st.warning("Please enter a search query")

if __name__ == "__main__":
    main()
