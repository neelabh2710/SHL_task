from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from app import retrieve_top_solutions, JobSolutionRetriever

app = FastAPI()

# Hardcode the CSV path as in your streamlit app
csv_path = r"Final_data_with_details.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at path: {csv_path}. Please place 'Final_data_with_details.csv' in the same folder.")

# Initialize the retriever once at startup
retriever = JobSolutionRetriever(csv_path)

# Define the request model
class SearchRequest(BaseModel):
    main_query: str

@app.post("/search")
def search_endpoint(request: SearchRequest):
    try:
        results = retrieve_top_solutions(
            main_query=request.main_query,
            retriever=retriever,
            num_subqueries=5,
            top_k_per_subquery=5,
            final_top_k=10
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
