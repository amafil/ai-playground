from sklearn.neighbors import NearestNeighbors


class CosineSimilarityIndex:
    def __init__(self, index: NearestNeighbors, faq_answers: dict[int, str]):
        self.index = index
        self.faq_answers = faq_answers
