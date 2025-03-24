import faiss
from models.knowledge import Knowledge


class EuclideanSimilarityIndex:
    def __init__(self, index: faiss.IndexFlatL2, faq_answers: dict[int, Knowledge]):
        self.index = index
        self.faq_answers = faq_answers
