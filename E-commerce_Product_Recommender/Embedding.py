import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

def get_embeddings():
    csv_path = r"D:\MCP SERVER PROJECTS\E-commerce_Product_Recommender\flipkart_com-ecommerce_sample.csv"
    index_path = r"D:\MCP SERVER PROJECTS\E-commerce_Product_Recommender\faiss_index"

    os.makedirs(index_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {len(df)} rows")

    docs = []
    for _, row in df.iterrows():
        try:
            desc = row['description'] if pd.notna(row['description']) else ""
            desc = desc[:200] 
            content = f"{row['product_name']} - {desc} - ₹{row['retail_price']}"
            docs.append(Document(page_content=content, metadata={"id": row['uniq_id']}))
        except Exception as e:
            print("Skipping row due to error:", e)

    print(f"Total documents to embed: {len(docs)}")

    if not docs:
        print("No documents found. Please check your CSV and columns.")
        return

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key= "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" # Replace with your actual key
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    print("FAISS index saved to:", index_path)

    print("Files in index_path:", os.listdir(index_path))

if __name__ == "__main__":
    get_embeddings()
