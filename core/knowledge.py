from typing import List
import os
from models.question_answer import QuestionAnswer


def load_knowledge_from_file(directory_path) -> List[QuestionAnswer]:
    knowledge: List[QuestionAnswer] = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            knowledge.append(QuestionAnswer.load(file_path))

    return knowledge
