from typing import List
import os

from models.question_answer import QuestionAnswer


def load_knowledge(directory_path) -> List[QuestionAnswer]:
    knowledge: List[QuestionAnswer] = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            knowledge.append(QuestionAnswer.load(file_path))

    return knowledge


# Load knowledge from the external file
knowledge_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
knowledge_directory = os.path.join(knowledge_directory, "QA")

# The `knowledge` variable contains all the knowledge about the helpdesk task.
# It is structured as a list of tuples, where each tuple represents a question and its corresponding answer.
# This data is used by FAISS (Facebook AI Similarity Search) for efficient similarity-based retrieval.

knowledge = load_knowledge(directory_path=knowledge_directory)
