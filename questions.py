import json
import numpy as np
from openai import OpenAI
from functions import generate_embedding


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


def chat_completion(message: str, knowledge: str, open_ai_client: OpenAI, model: str):
    chat = [
        {
            "role": "system",
            "content": (
                "You are a helpdesk bot that assists users by automatically responding to support requests via email."
                " Always respond in Italian. If a user writes in English, inform them that you can only respond in Italian."
                " Your responses must be strictly based on the provided FAQ data."
                " If the question is not covered in the FAQ fata, respond with:"
                " *\"I'm sorry, but I don't have an answer.*."
                " Important guidelines:"
                " 1. **Use a polite and clear tone**."
                " 2. **Do not generate responses or add information that is not in the FAQ**."
                " 3. **Follow formatting rules**: Use bullet points and emojis only if they are already present in the FAQ."
                " 4. **Include links only if they are explicitly mentioned in the FAQ**."
                " 5. **If the user requests additional details not found in the FAQ, suggest contacting official support**."
            ),
        },
        {
            "role": "user",
            "content": f"The following is the available FAQ data:\n\n{knowledge}\n\n"
            "Use only this information to answer user queries.",
        },
        {
            "role": "user",
            "content": f"Generate a response based on the following message:\n\n{message}",
        },
    ]

    print("DEBUG chat =>>", json.dumps(chat, indent=2))

    response = open_ai_client.chat.completions.create(model=model, messages=chat)

    return response.choices[0].message.content
