class Question:
    def __init__(self, question_id, question_text):
        self.question_id = question_id
        self.question_text = question_text

    def __str__(self):
        return f"{self.question_id} {self.question_text}"