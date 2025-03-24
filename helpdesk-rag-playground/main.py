import os
from typing import List
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from models.cosine_similarity_index import CosineSimilarityIndex
from models.euclidean_similarity_index import EuclideanSimilarityIndex
from models.knowledge import Knowledge
from models.vectorial_search_result import VectorialSearchResult
from services.search import (
    search_with_cosine_similarity,
    search_with_euclidean_similarity,
)
from services.chat import chat_completion
from core.embeddings import (
    init_index_for_cosine_similarity,
    init_index_for_euclidean_similarity,
)
from core.classification import classify_message, extract_keyword, extract_keyword_llm
from core.knowledge import load_knowledge_from_file
from core.utils import (
    mock_helpdesk_messages,
)

if __name__ == "__main__":
    chat_client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    )
    CHAT_COMPLETION_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-all-minilm-l6-v2-embedding"

    # Load knowledge from the external files
    knowledge_directory = os.path.join(os.path.dirname(__file__), "data")
    knowledge_directory = os.path.join(knowledge_directory, "QA")

    # The `knowledge` variable contains all the knowledge about the helpdesk task.
    knowledge: List[Knowledge] = load_knowledge_from_file(
        directory_path=knowledge_directory
    )

    messages = mock_helpdesk_messages()

    if not messages:
        print("No new messages found")
        exit()
    else:
        print(f"Found {len(messages)} messages")

    cosine_similarity_index: CosineSimilarityIndex = init_index_for_cosine_similarity(
        knowledge=knowledge,
        open_ai_client=chat_client,
        embedding_model=EMBEDDING_MODEL,
    )

    euclidean_similarity_index: EuclideanSimilarityIndex = (
        init_index_for_euclidean_similarity(
            knowledge=knowledge,
            open_ai_client=chat_client,
            embedding_model=EMBEDDING_MODEL,
        )
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

            cosine_similarity_based_answers: List[VectorialSearchResult] = (
                search_with_cosine_similarity(
                    question=trimmed_message,
                    index=cosine_similarity_index.index,
                    open_ai_client=chat_client,
                    embedding_model=EMBEDDING_MODEL,
                    faq_answers=cosine_similarity_index.faq_answers,
                )
            )

            euclidean_similarity_based_answers: List[VectorialSearchResult] = (
                search_with_euclidean_similarity(
                    question=trimmed_message,
                    index=euclidean_similarity_index.index,
                    open_ai_client=chat_client,
                    embedding_model=EMBEDDING_MODEL,
                    faq_answers=euclidean_similarity_index.faq_answers,
                )
            )

            if (
                not cosine_similarity_based_answers
                and not euclidean_similarity_based_answers
            ):
                print("No answers found")
                continue

            llm_knowledge: List[VectorialSearchResult] = []
            print("=" * 40, "\033[96mSearch results\033[0m", "=" * 40)
            for search_result in cosine_similarity_based_answers:
                print("\033[93m*\033[0m Cosine", search_result)
                # if search_result.index is in euclidean_similarity_based_answers then skip to avoid duplicates
                if search_result.index in [
                    result.index for result in euclidean_similarity_based_answers
                ]:
                    continue
                llm_knowledge.append(search_result)

            for search_result in euclidean_similarity_based_answers:
                print("\033[94m*\033[0m Euclidean", search_result)
                llm_knowledge.append(search_result)

            print("=" * 80)

            chat = chat_completion(
                message=trimmed_message,
                knowledge=llm_knowledge,
                open_ai_client=chat_client,
                model=CHAT_COMPLETION_MODEL,
            )

            print(f"\033[98m{chat}\033[0m")
