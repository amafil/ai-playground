import os
from openai import OpenAI
from questions import chat_completion, vectorial_answer_search
from database import init_index
from message_classificator import classify_message, extract_keyword
from functions import (
    get_helpdesk_messages,
)

if __name__ == "__main__":
    chat_client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.1.101:1234/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    )
    CHAT_COMPLETION_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-all-minilm-l6-v2-embedding"

    messages = get_helpdesk_messages()

    if not messages:
        print("No new messages found")
        exit()
    else:
        print(f"Found {messages.count} messages")

    index, knowledge = init_index(chat_client, EMBEDDING_MODEL)
    for message in messages:
        if message:
            trimmed_message = message.strip()

            print("Seeking an answer for:", trimmed_message)

            message_classification = classify_message(
                message=trimmed_message,
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            message_keyworkds = extract_keyword(
                message=trimmed_message,
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            answers = vectorial_answer_search(
                question=trimmed_message,
                embedding_model=EMBEDDING_MODEL,
                open_ai_client=chat_client,
                index=index,
                knowledge=knowledge,
            )

            if not answers:
                print("No answer found")
                continue

            print(answers)
            chat = chat_completion(
                message=trimmed_message,
                knowledge=answers,
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            print("Chat completion:\n", chat)
