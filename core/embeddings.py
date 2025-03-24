from typing import List
import faiss
from openai import OpenAI
import numpy as np
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from models.cosine_similarity_index import CosineSimilarityIndex
from models.euclidean_similarity_index import EuclideanSimilarityIndex
from models.knowledge import Knowledge


def generate_embedding(question: str, open_ai_client: OpenAI, model: str) -> np.ndarray:
    response = open_ai_client.embeddings.create(model=model, input=question)
    embedding = np.array(
        response.data[0].embedding, dtype=np.float32
    )  # Convert to NumPy array

    return embedding


def init_index_for_cosine_similarity(
    knowledge: List[Knowledge], open_ai_client: OpenAI, embedding_model: str
) -> CosineSimilarityIndex:
    print("Initializing index for cosine similarity...")

    faq_embeddings = np.array(
        [
            generate_embedding(qa.question, open_ai_client, embedding_model)
            for qa in knowledge
        ],
        dtype=np.float32,
    )

    # Use Nearest Neighbors with cosine similarity
    index = NearestNeighbors(n_neighbors=1, metric="cosine")
    index.fit(faq_embeddings)

    faq_answers = {i: knowledge[i].answer for i in range(len(knowledge))}

    return CosineSimilarityIndex(index, faq_answers)


def init_index_for_euclidean_similarity(
    knowledge: List[Knowledge], open_ai_client: OpenAI, embedding_model: str
) -> EuclideanSimilarityIndex:
    print("Initializing index for euclidean similarity...")

    dimension = 384  # Match the embedding model output
    index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) similarity

    print("Generating Knowledge embeddings...")

    # Store embeddings and metadata
    faq_embeddings: np.ndarray = np.array(
        [
            generate_embedding(
                question=k.question,
                open_ai_client=open_ai_client,
                model=embedding_model,
            )
            for k in knowledge
        ],
        dtype=np.float32,
    )

    print("Indexing Knowledge embeddings...")

    index.add(faq_embeddings)
    faq_answers: dict[int, Knowledge] = {
        i: k for i, k in enumerate(knowledge)  # Map index to Knowledge object
    }  # Store index-answer mapping

    return EuclideanSimilarityIndex(index, faq_answers)
