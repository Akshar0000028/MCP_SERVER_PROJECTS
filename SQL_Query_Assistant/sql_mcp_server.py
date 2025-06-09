from fastmcp import FastMCP
import sqlite3
from fastapi import FastAPI, Request

mcp = FastMCP()
api = FastAPI()


def execute_sql(query: str) -> dict:
    """
    Executes a SQL query on data.db and returns rows and columns.
    """
    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}


@api.post("/query")
async def query_db(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            return {"error": "Missing 'query' field in request body"}
        result = execute_sql(query)  
        return result
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

app = api
