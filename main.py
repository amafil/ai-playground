import os
from typing import List
from openai import OpenAI
from models.question_answer import QuestionAnswer
from services.search import vectorial_answer_search
from services.chat import chat_completion
from core.embeddings import init_index
from core.classification import classify_message, extract_keyword, extract_keyword_llm
from core.knowledge import load_knowledge_from_file
from core.utils import (
    mock_helpdesk_messages,
)

if __name__ == "__main__":
    chat_client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://192.168.1.101:1234/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    )
    CHAT_COMPLETION_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-all-minilm-l6-v2-embedding"

    # Load knowledge from the external files
    knowledge_directory = os.path.join(os.path.dirname(__file__), "data")
    knowledge_directory = os.path.join(knowledge_directory, "QA")

    # The `knowledge` variable contains all the knowledge about the helpdesk task.
    knowledge: List[QuestionAnswer] = load_knowledge_from_file(directory_path=knowledge_directory)

    messages = mock_helpdesk_messages()

    if not messages:
        print("No new messages found")
        exit()
    else:
        print(f"Found {len(messages)} messages")

    index, faq_embeddings, faq_answers = init_index(
        knowledge=knowledge,
        open_ai_client=chat_client,
        embedding_model=EMBEDDING_MODEL,
    )
    for message in messages:
        if message:
            trimmed_message = message.strip()

            print("Seeking an answer for:", trimmed_message)

            message_classification = classify_message(
                message=trimmed_message,
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            message_keyworkds_llm = extract_keyword_llm(
                message=trimmed_message,
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            message_keyworkds = extract_keyword(message=trimmed_message)

            answers = vectorial_answer_search(
                question=trimmed_message,
                index=index,
                open_ai_client=chat_client,
                embedding_model=EMBEDDING_MODEL,
                faq_answers=faq_answers,
            )

            if not answers:
                print("No answer found")
                continue

            print(answers)
            chat = chat_completion(
                message=trimmed_message,
                knowledge=answers, #FIXME: this is a List[QuestionAnswer] not a string
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            print("Chat completion:\n", chat)
