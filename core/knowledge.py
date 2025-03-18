import os


def load_knowledge(directory_path):
    knowledge = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                print(f"Loading knowledge from {file_path}")
                content = file.read().strip()
                entries = content.split(
                    "\n\n"
                )  # Separate questions and answers by two newlines

                for entry in entries:
                    question, answer = entries[0], entries[1]
                    knowledge.append((question.strip(), answer.strip()))

    return knowledge


# Load knowledge from the external file
knowledge_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
knowledge_directory = os.path.join(knowledge_directory, "QA")

# The `knowledge` variable contains all the knowledge about the helpdesk task.
# It is structured as a list of tuples, where each tuple represents a question and its corresponding answer.
# This data is used by FAISS (Facebook AI Similarity Search) for efficient similarity-based retrieval.

knowledge = load_knowledge(directory_path=knowledge_directory)
