class SemanticLogitsProcessor(LogitsProcessor):
    def __init__(self, reference_text: str, encoder, threshold: float = 0.65):
        # Esto se ejecuta UNA vez por sesión, no por token
        self.reference_embedding = encoder.encode(reference_text)
        self.threshold = threshold

    def __call__(self, input_ids, scores):
        # Aquí NO se toca la red de embeddings
        ...