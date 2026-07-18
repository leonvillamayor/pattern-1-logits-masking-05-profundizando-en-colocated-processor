if all(score < self.threshold for score in valid_scores):
    raise AllCandidatesBelowThreshold(
        f"Ningún candidato superó {self.threshold} tras {len(valid_scores)} intentos"
    )