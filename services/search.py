import numpy as np
from core.embeddings import generate_embedding


def vectorial_answer_search(
    question,
    embedding_model,
    open_ai_client,
    index,
    knowledge,
    top_k=3,
    distance_threshold=0.5,
):
    query_embedding = generate_embedding(
        question=question,
        open_ai_client=open_ai_client,
        model=embedding_model,
    )

    numpy_embedding = np.array(query_embedding).reshape(1, -1)  # Convert to NumPy array

    distances, indices = index.search(numpy_embedding, top_k)

    results = [
        (knowledge[i], distances[0][j])
        for j, i in enumerate(indices[0])
        if distances[0][j] < distance_threshold
    ]
    return results
