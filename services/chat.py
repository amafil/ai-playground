from openai import OpenAI


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

    response = open_ai_client.chat.completions.create(model=model, messages=chat)

    return response.choices[0].message.content
