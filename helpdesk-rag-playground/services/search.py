from typing import List
import faiss
import numpy as np
from openai import OpenAI
from core.embeddings import generate_embedding
from sklearn.neighbors import NearestNeighbors
from models.vectorial_search_result import VectorialSearchResult


def search_with_cosine_similarity(
    question: str,
    index: NearestNeighbors,
    open_ai_client: OpenAI,
    embedding_model: str,
    faq_answers: dict[int, str],
    n_neighbors: int = 3,
    distance_threshold: float = 0.3,
) -> List[VectorialSearchResult]:
    query_embedding = generate_embedding(
        question, open_ai_client, embedding_model
    ).reshape(1, -1)

    scores, indices = index.kneighbors(X=query_embedding, n_neighbors=n_neighbors)

    result: List[VectorialSearchResult] = []
    for i in range(n_neighbors):
        if scores[0][i] > distance_threshold:
            continue

        if indices[0][i] in faq_answers:
            result.append(
                VectorialSearchResult(
                    scores[0][i], indices[0][i], faq_answers[indices[0][i]]
                )
            )

    return result


def search_with_euclidean_similarity(
    question: str,
    embedding_model: str,
    open_ai_client: str,
    index: faiss.IndexFlatL2,
    faq_answers: List[str],
    top_k=3,
    distance_threshold=0.5,
) -> List[VectorialSearchResult]:
    query_embedding = generate_embedding(
        question=question,
        open_ai_client=open_ai_client,
        model=embedding_model,
    )

    numpy_embedding = np.array(query_embedding).reshape(1, -1)  # Convert to NumPy array

    distances, indices = index.search(numpy_embedding, top_k)

    result: List[VectorialSearchResult] = []
    for i in range(top_k):
        if distances[0][i] > distance_threshold:
            continue

        if indices[0][i] in faq_answers:
            result.append(
                VectorialSearchResult(
                    distances[0][i], indices[0][i], faq_answers[indices[0][i]].answer
                )
            )

    return result
