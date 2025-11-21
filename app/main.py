from typing import Union
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from helper_lib.model import load_model

from app.bigram_model import BigramModel
from app.embeddings import (
    calculate_embedding,
    calculate_similarity,
    linear_algebra_similarity,
)
from app.predict_cnn import predict_cnn
from app.predict_gan import generate_gan_image
from app.predict_ebm import router as ebm_router
from app.predict_diffusion import router as diffusion_router
from app.predict_rnn import load_or_train_rnn_model, generate_rnn_text
from app.predict_llm import load_or_train_llm_model, generate_llm_text, FORMAT_PREFIX, FORMAT_SUFFIX


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

# RNN (LSTM) language model - Module 6 with checkpoints
rnn_device, rnn_model, rnn_vocab, rnn_inv_vocab = load_or_train_rnn_model()

# Fine-tuned GPT-2 LLM - Module 9 activity
llm_device, llm_model, llm_tokenizer = load_or_train_llm_model()

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

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    """
    RNN-based (LSTM) text generation using the Module 6 architecture
    trained on 'The Count of Monte Cristo'.
    """
    generated_text = generate_rnn_text(
        model=rnn_model,
        vocab=rnn_vocab,
        inv_vocab=rnn_inv_vocab,
        seed_text=request.start_word,
        length=request.length,
        temperature=1.0,   
        dev=rnn_device,
    )
    return {"generated_text": generated_text}

@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    """
    Text generation using the fine-tuned GPT-2 model on SQuAD.

    The *final output format* is:
      "That is a great question ... let me know if you have any other questions"
    """
    prompt = request.start_word

    # Raw model output (QA-style)
    raw = generate_llm_text(
        model=llm_model,
        tokenizer=llm_tokenizer,
        prompt=prompt,
        max_new_tokens=request.length,
        temperature=0.8,
        top_p=0.9,
        dev=llm_device,
        raw_only=True,
    )

    # Enforce exact prefix & suffix required by the assignment
    formatted = (
        f"{FORMAT_PREFIX}, "
        + raw
        + f" {FORMAT_SUFFIX}"
    )

    return {"generated_text": formatted}


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

@app.post("/predict_cnn")
async def predict_route(file: UploadFile = File(...)):
    """Call the CNN prediction function from external module."""
    return await predict_cnn(file)


@app.post("/generate_gan")
async def generate_gan(num_images: int = 1):
    """
    Generate synthetic images using the trained GAN model.
    """
    return await generate_gan_image(num_images)

# EBM Endpoint
app.include_router(ebm_router, prefix="/ebm")

# Diffusion Endpoint
app.include_router(diffusion_router, prefix="/diffusion")