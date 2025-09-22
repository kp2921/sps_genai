from functools import lru_cache
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

@lru_cache(maxsize=1)
def _nlp():
    # Mirrors your notebook's model choice
    # Make sure you've run: uv run python -m spacy download en_core_web_lg
    return spacy.load("en_core_web_lg")

def calculate_embedding(input_word: str) -> list[float]:
    doc = _nlp()(input_word)
    return doc.vector.tolist()

def calculate_similarity(word1: str, word2: str) -> float:
    nlp = _nlp()
    return float(nlp(word1).similarity(nlp(word2)))

def linear_algebra_similarity(word1: str, word2: str, word3: str, word4: str) -> float:
    """
    Recreates the 'word1 + word2 - word3' vs 'word4' cosine similarity
    """
    nlp = _nlp()
    v1 = nlp(word1).vector
    v2 = nlp(word2).vector
    v3 = nlp(word3).vector
    target = nlp(word4).vector
    composed = v1 + (v2 - v3)

    # keep sklearn usage to match your notebook
    return float(sk_cosine_similarity([composed], [target])[0][0])
