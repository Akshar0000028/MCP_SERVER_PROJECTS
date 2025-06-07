import requests
from typing import List
import sys

class RecipeRecommenderClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize with either MCP client or fallback to HTTP
        """
        self.base_url = base_url
        self.use_mcp = False
        
        
    
    def recommend_recipes(self, ingredients: str, top_k: int = 5) -> List[str]:
        """
        Get recommendations using best available method
        """
        if self.use_mcp:
            return self._recommend_via_mcp(ingredients, top_k)
        return self._recommend_via_http(ingredients, top_k)
    
    def _recommend_via_http(self, ingredients: str, top_k: int) -> List[str]:
        """HTTP API implementation"""
        response = requests.post(
            f"{self.base_url}/recommend",
            json={"ingredients": ingredients, "top_k": top_k}
        )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"HTTP Error: {response.status_code} - {response.text}")
    
    def _recommend_via_mcp(self, ingredients: str, top_k: int) -> List[str]:
        """MCP client implementation"""
        try:
            return self.mcp_client.RecipeRecommender(ingredients=ingredients, top_k=top_k)
        except Exception as e:
            print(f"MCP Error: {e}, falling back to HTTP")
            self.use_mcp = False
            return self._recommend_via_http(ingredients, top_k)

def main():
    client = RecipeRecommenderClient()
    
    print("Recipe Recommendation System")
    
    
    while True:
        try:
            ingredients = input("\nEnter ingredients: ").strip()
            if not ingredients:
                print("Please enter ingredients")
                continue
                
            top_k = input("Number of recommendations [5]: ").strip()
            top_k = int(top_k) if top_k else 5
            
            recipes = client.recommend_recipes(ingredients, top_k)
            
            print(f"\nTop {len(recipes)} recommendations:")
            for i, recipe in enumerate(recipes, 1):
                print(f"{i}. {recipe}")
                
            if input("\nContinue? (y/n): ").lower() != 'y':
                break
                
        except ValueError:
            print("Please enter a valid number")
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()