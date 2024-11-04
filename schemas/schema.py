from fastapi import  HTTPException, Query
from pydantic import BaseModel



class QueryResponse(BaseModel):
    content: str
    similarity_score: float
