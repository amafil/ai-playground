class VectorialSearchResult:
    def __init__(self, score: float, index: int, answer: str):
        self.score = score
        self.index = index
        self.answer = answer

    def __str__(self):
        return f"Score: {self.score} Index: {self.index}, Answer: {self.answer}"
