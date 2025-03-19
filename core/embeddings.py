from typing import List
from openai import OpenAI
import numpy as np
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from models.knowledge import Knowledge


def generate_embedding(question: str, open_ai_client: OpenAI, model: str):
    response = open_ai_client.embeddings.create(model=model, input=question)
    embedding = np.array(
        response.data[0].embedding, dtype=np.float32
    )  # Convert to NumPy array

    return embedding


def init_index(
    knowledge: List[Knowledge], open_ai_client: OpenAI, embedding_model: str
):
    print("Initializing index...")

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

    return index, faq_embeddings, faq_answers
