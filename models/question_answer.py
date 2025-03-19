class QuestionAnswer:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def __str__(self):
        return f"{self.question} {self.answer}"

    def load(file_path) -> "QuestionAnswer":
        result: QuestionAnswer = None
        with open(file_path, "r", encoding="utf-8") as file:
            print(f"Loading knowledge from {file_path}")
            content = file.read().strip()
            entries = content.split(
                "\n\n"
            )  # Separate questions and answers by two newlines

            result = QuestionAnswer(entries[0], entries[1])

        return result
