import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

df = pd.read_csv("D:\\MCP SERVER PROJECTS\\Receipe_Recommendation\\recipe_final (1).csv")

def preprocess_ingredients(row):
    ingredients = row['ingredients_list']
    if isinstance(ingredients, str):
        return ingredients.lower()
    return ""

df['processed_ingredients'] = df.apply(preprocess_ingredients, axis=1)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['processed_ingredients'].tolist(), convert_to_numpy=True)

faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, 'recipe_faiss.index')
df[['recipe_name', 'processed_ingredients']].to_csv('recipe_metadata.csv', index=False)

print("successfully!")
