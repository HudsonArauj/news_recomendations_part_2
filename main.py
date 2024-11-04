from fastapi import FastAPI, HTTPException, Query
from scripts.get_data import get_data
from schemas.schema import QueryResponse
import faiss
from scripts.preprocess import search
import os
import requests
from io import BytesIO

import torch

torch.set_num_threads(1) 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

index = faiss.read_index("vectordb_news.faiss")
df = get_data() 

app = FastAPI()

@app.get("/query")
def query_route(query: str = Query(..., description="Search query")):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    similar_texts, similarity_scores = search(query, index, df['texto'].tolist(), similarity_threshold=0.6)
    
    results = []
    for i, text in enumerate(similar_texts):
        results.append(QueryResponse(content=text, similarity_score=similarity_scores[i]))

    return {"results": results, "message": "OK", "n_results": len(results)}



@app.post("/update_model")
def update_model(model_url: str):
    global index
    try:
        response = requests.get(model_url)
        response.raise_for_status() 
        model_buffer = BytesIO(response.content)
        
        index = faiss.read_index_binary(model_buffer.getvalue())
        
        return {"status": "Índice atualizado com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar o índice: {str(e)}")
