from typing import List

class VectorialSearchResult:
    def __init__(self, score: float, index: int, answers: List[str]):
        self.score = score
        self.index = index
        self.answers = answers

    def __str__(self):
        return f"Score: {self.score} Index: {self.index}, Answer: {self.answers}"
