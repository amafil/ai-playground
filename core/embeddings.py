from openai import OpenAI
import faiss
import numpy as np
from openai import OpenAI
from core.knowledge import knowledge


def generate_embedding(question: str, open_ai_client: OpenAI, model: str):
    response = open_ai_client.embeddings.create(model=model, input=question)
    embedding = response.data[0].embedding

    return embedding


def init_index(open_ai_client: OpenAI, embedding_model: str):
    print("Initializing index...")

    dimension = 384  # Match the embedding model output
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) similarity

    print("Generating FAQ embeddings...")

    # Store embeddings and metadata
    faq_embeddings = np.array(
        [
            generate_embedding(
                question=q, open_ai_client=open_ai_client, model=embedding_model
            )
            for q, _ in knowledge
        ],
        dtype=np.float32,
    )

    print("Indexing FAQ embeddings...")

    index.add(faq_embeddings)
    faq_answers = {
        i: knowledge[i][1] for i in range(len(knowledge))
    }  # Store index-answer mapping

    return index, faq_answers
