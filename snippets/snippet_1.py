"""
Episode 5 — Colocated Processor (deep dive)
La cuenta que sale cara: round-trips por token y percentil 99.

Same LogitsProcessor signature as eps. 3 & 4. Same SemanticSimilarityScorer
contract (ref embedding precomputed outside __call__, threshold=0.65). We
only add instrumentation to answer: "calls/token x p99 latency — does
colocation still pay off, or did we just move the bottleneck?"
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Reference text and scorer contract (as established in eps. 3 & 4)
REF_TEXT = "el paciente no presentó reacciones adversas tras la dosis"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class SemanticSimilarityScorer:
    """Same contract as eps. 3 & 4 — colocated, low-latency."""

    def __init__(self, ref_text: str, model_name: str, threshold: float = 0.65):
        self.model = SentenceTransformer(model_name)
        # Reference embedding precomputed ONCE, outside the hot path.
        self.ref = self.model.encode([ref_text], normalize_embeddings=True)[0]
        self.threshold = threshold

    def score(self, candidate_text: str) -> float:
        v = self.model.encode([candidate_text], normalize_embeddings=True)[0]
        return cosine(self.ref, v)

    def passes(self, candidate_text: str) -> bool:
        return self.score(candidate_text) >= self.threshold


# --- Episode 5 novelty: latency + call-count instrumentation ----------
@dataclass
class CallStats:
    calls: int = 0
    tokens: int = 0
    latencies_us: list[int] = field(default_factory=list)

    def record(self, latency_us: int) -> None:
        self.calls += 1
        self.latencies_us.append(latency_us)

    def note_token(self) -> None:
        self.tokens += 1

    def pct(self, p: int) -> float:
        if not self.latencies_us:
            return 0.0
        # 99 cut points for n=100 inclusive quantiles; index = p - 1.
        cut = statistics.quantiles(self.latencies_us, n=100, method="inclusive")
        return cut[p - 1]

    def report(self, colocated: bool) -> str:
        lus = self.latencies_us
        if not lus:
            return "no calls recorded"
        cpt = self.calls / max(self.tokens, 1)
        total_ms = sum(lus) / 1000.0
        return (
            f"\n=== Latency report ({'COLOCATED' if colocated else 'REMOTE'}) ===\n"
            f"  tokens generated         : {self.tokens}\n"
            f"  scorer calls             : {self.calls}\n"
            f"  calls / token            : {cpt:.2f}\n"
            f"  total scorer time        : {total_ms:,.1f} ms\n"
            f"  p50  latency             : {self.pct(50):>8.1f} µs/call\n"
            f"  p95  latency             : {self.pct(95):>8.1f} µs/call\n"
            f"  p99  latency             : {self.pct(99):>8.1f} µs/call\n"
            f"  max                      : {max(lus):>8.1f} µs/call\n"
            f"  budget per token (p99)   : {cpt * self.pct(99):>8.1f} µs\n"
        )


class InstrumentedScorer:
    """Wraps any scorer; same call signature, just times it."""

    def __init__(self, inner: SemanticSimilarityScorer, stats: CallStats, colocated: bool):
        self.inner = inner
        self.stats = stats
        # Network RTT added only in the REMOTE case.
        # Colocated rack: tens of µs (we model as ~0 on top of encode()).
        # Cross-cloud embedding API: tens of ms per round-trip.
        self.network_overhead_us = 0 if colocated else 30_000

    def score(self, candidate_text: str) -> float:
        t0 = time.perf_counter_ns()
        s = self.inner.score(candidate_text)
        elapsed_us = (time.perf_counter_ns() - t0) // 1000 + self.network_overhead_us
        self.stats.record(elapsed_us)
        return s


# --- Mimics the inner loop of a LogitsProcessor / HF generate() -------
# For each generated token the processor fans out over CANDIDATES_PER_TOKEN
# candidates and asks the scorer for each one. Real vocab fan-out is much
# larger; we keep it small so the demo runs in seconds, the metric is the
# *ratio* calls/token and the latency percentiles per call.

CANDIDATES_PER_TOKEN = 32


def simulate_generation(scorer: InstrumentedScorer, stats: CallStats, n_tokens: int) -> int:
    rng = np.random.default_rng(0)
    survivors = 0
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")
    for _ in range(n_tokens):
        for _ in range(CANDIDATES_PER_TOKEN):
            cand = REF_TEXT + " " + "".join(rng.choice(alphabet, 8))
            if scorer.score(cand) >= 0.65:
                survivors += 1
        stats.note_token()
    return survivors


# --- Run both scenarios and compare ----------------------------------
if __name__ == "__main__":
    n_tokens = 200  # short generation, enough for stable percentiles

    print("Loading all-MiniLM-L6-v2 (local, free) — same model as eps. 3 & 4…")
    base = SemanticSimilarityScorer(
        REF_TEXT, "sentence-transformers/all-MiniLM-L6-v2", threshold=0.65
    )

    for colocated in (True, False):
        stats = CallStats()
        wrapped = InstrumentedScorer(base, stats, colocated=colocated)
        _ = simulate_generation(wrapped, stats, n_tokens)
        print(stats.report(colocated))

    print(
        "Take-away: con el scorer COLOCADO el p99 se queda en cientos de µs y\n"
        "           el presupuesto por token (calls/token x p99) cabe holgado\n"
        "           en el step-time del LM. En cuanto el scorer sale del rack,\n"
        "           ese mismo producto revienta el budget aunque el modelo\n"
        "           subyacente sea idéntico. La 'cuenta que sale cara' es,\n"
        "           siempre, calls/token x latencia.\n"
        "Bridge     → Próximo ep.: 'Profundizando en Proxy Metric'. Tener un\n"
        "           p99 bonito no es lo mismo que tener un scorer útil: cuándo\n"
        "           ese p99 es proxy honesto de calidad y cuándo te está\n"
        "           mintiendo (especialmente en los textos <20 tokens y en\n"
        "           las 4 trampas del ep. 4)."
    )