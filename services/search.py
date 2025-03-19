from typing import List
from openai import OpenAI
from core.embeddings import generate_embedding
from sklearn.neighbors import NearestNeighbors
from models.vectorial_search_result import VectorialSearchResult


def vectorial_answer_search(
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
        if(scores[0][i] > distance_threshold):
            continue

        if indices[0][i] in faq_answers:
            result.append(VectorialSearchResult(scores[0][i], indices[0][i], faq_answers[indices[0][i]]))

    return result
