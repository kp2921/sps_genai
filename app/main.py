from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embeddings import (
    calculate_embedding,
    calculate_similarity,
    linear_algebra_similarity,
)
import numpy as np

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    query_word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class LASimilarityRequest(BaseModel):
    word1: str  
    word2: str  
    word3: str  
    word4: str  

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(req: EmbeddingRequest):
    vec = calculate_embedding(req.query_word)
    return {"word": req.query_word, "embedding": vec, "dim": len(vec)}

@app.post("/similarity")
def get_similarity(req: SimilarityRequest):
    sim = calculate_similarity(req.word1, req.word2)
    return {"word1": req.word1, "word2": req.word2, "similarity": sim}

@app.post("/la_similarity")
def get_la_similarity(req: LASimilarityRequest):
    """
    Computes cosine similarity between:
      (word1 + word2 - word3)  and  word4
    """
    sim = linear_algebra_similarity(req.word1, req.word2, req.word3, req.word4)
    return {
        "expression": f"{req.word1} + {req.word2} - {req.word3}",
        "compare_to": req.word4,
        "cosine_similarity": sim,
    }

#@app.get("/gaussian")
#def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> List[float]:
#    """Samples from a Gaussian distribution with given mean and variance."""
#    std_dev = np.sqrt(variance)
#    samples = np.random.normal(mean, std_dev, size)
#    return samples.tolist()


