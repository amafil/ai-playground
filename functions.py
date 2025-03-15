from openai import OpenAI


def generate_embedding(question: str, open_ai_client: OpenAI, model: str):
    response = open_ai_client.embeddings.create(model=model, input=question)
    embedding = response.data[0].embedding

    return embedding


def get_helpdesk_messages():
    result = []
    # This is a mock function. Replace it with the actual implementation
    # of the function that retrieves messages from the helpdesk channel.
    result.append(
        "here is the first request mock from the user. Use this method to retrieve the messages from the helpdesk channel."
    )
    return result
