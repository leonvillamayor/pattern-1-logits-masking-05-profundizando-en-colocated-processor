def score_candidate(self, candidate_text: str) -> float:
    cand_emb = self.encoder.encode(candidate_text)
    return cosine_similarity(self.reference_embedding, cand_emb)

def __call__(self, input_ids, scores):
    for i, token_id in enumerate(top_candidates):
        text = self.tokenizer.decode(token_id)
        if self.score_candidate(text) < self.threshold:
            scores[i] = float("-inf")  # anulamos este candidato
    return scores