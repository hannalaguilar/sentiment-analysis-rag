from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from sentiment_llm import classify_rag_faiss

# FastAPI app
app = FastAPI(title='Sentiment Analysis API', description='This is a API for serve a LLM model', version='0.0.1')

# Schemas
class ReviewRequest(BaseModel):
    review_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "review_text": "This movie was absolutely fantastic!"
            }
        }


class SimilarReview(BaseModel):
    review: str
    sentiment: str
    cosine_score: float


class ReviewResponse(BaseModel):
    sentiment: str
    similar_reviews: List[SimilarReview]

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "similar_reviews": [
                    {
                        "review": "An emotional, gripping movie with great performances.",
                        "sentiment": "positive",
                        "cosine_score": 0.8124
                    }
                ]
            }
        }


# Endpoint
@app.post("/classify", response_model=ReviewResponse)
def classify(request: ReviewRequest):
    result = classify_rag_faiss(request.review_text, verbose=False)
    return ReviewResponse(**result)
