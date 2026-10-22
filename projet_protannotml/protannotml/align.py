"""Exported from notebook pf03_baselines.ipynb — edit the notebook source (src/pf03_baselines.md), not this file."""

"""Alignment-based annotation baseline (germline alignment + IMGT rules), in the spirit of IMGT/DomainGapAlign."""
import numpy as np

from protannotml.imgt import cdr_positions, region_of, pos_key

AA = "ACDEFGHIKLMNPQRSTVWY*"
GROUPS = ["ILMV", "FWY", "KRH", "DE", "NQ", "ST", "AG"]
SCORE = np.full((22, 22), -1.0)
for _a in range(21):
    for _b in range(21):
        if _a == _b:
            SCORE[_a, _b] = 4.0
        elif any(AA[_a] in g and AA[_b] in g for g in GROUPS):
            SCORE[_a, _b] = 1.0
GAP = 4.0


def encode(seq):
    return np.array([AA.index(c) if c in AA else 21 for c in seq])


def overlap_align(query, ref, gap=GAP):
    """Alignment with free end gaps on both sequences (linear gaps). Returns (score, [(q_index, r_index)])."""
    qi, ri = encode(query), encode(ref)
    n, m = len(qi), len(ri)
    H = np.zeros((n + 1, m + 1))
    P = np.zeros((n + 1, m + 1), dtype=np.int8)            # 0 diagonal, 1 up (gap in ref), 2 left (gap in query)
    ar = np.arange(m + 1)
    for i in range(1, n + 1):
        diag = H[i - 1, :-1] + SCORE[qi[i - 1], ri]
        up = H[i - 1, 1:] - gap
        T = np.empty(m + 1)
        T[0] = 0.0
        T[1:] = np.maximum(diag, up)
        H[i] = np.maximum.accumulate(T + gap * ar) - gap * ar     # left moves via a prefix max
        P[i, 1:] = np.where(H[i, 1:] == diag, 0, np.where(H[i, 1:] == up, 1, 2))
    j_row, i_col = int(np.argmax(H[n])), int(np.argmax(H[:, m]))
    i, j = (n, j_row) if H[n, j_row] >= H[i_col, m] else (i_col, m)
    score, pairs = H[i, j], []
    while i > 0 and j > 0:
        if P[i, j] == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return float(score), pairs[::-1]


def kmers(seq, k=3):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


class GermlineAligner:
    """Annotate a V domain by aligning it to a reference directory of V and J germlines."""

    def __init__(self, v_refs, j_refs, top=3):
        self.v = [r for r in v_refs if r["functionality"] == "F"]
        self.v_pos = [sorted(r["positions"], key=pos_key) for r in self.v]
        self.v_seq = ["".join(r["positions"][p] for p in ps) for r, ps in zip(self.v, self.v_pos)]
        self.v_kmers = [kmers(s) for s in self.v_seq]
        self.j = [(r, "".join(r["positions"][str(p)] for p in range(118, 129))) for r in j_refs]
        self.top = top

    def best_germline(self, query):
        qk = kmers(query)
        jac = np.array([len(qk & k) / max(1, len(qk | k)) for k in self.v_kmers])
        candidates = np.argsort(-jac)[: self.top]
        results = [(overlap_align(query, self.v_seq[c]), c) for c in candidates]
        (score, pairs), c = max(results, key=lambda x: x[0][0])
        return c, score, pairs

    def annotate(self, query):
        c, score, pairs = self.best_germline(query)
        ref, ref_pos = self.v[c], self.v_pos[c]
        aligned = {qi: ref_pos[rj] for qi, rj in pairs}
        n = len(query)
        pos = [None] * n
        for qi, p in aligned.items():
            if region_of(p) in ("FR1", "FR2", "FR3"):
                pos[qi] = p
        fr_rows = lambda lo, hi: [qi for qi, p in aligned.items() if lo <= int(p) <= hi and region_of(p).startswith("FR")]

        def number_segment(region, first, last):
            seg = list(range(first, last))
            try:
                for qi, p in zip(seg, cdr_positions(region, len(seg))):
                    pos[qi] = p
            except ValueError:
                pass                                         # abnormal length: left unnumbered (counted as errors)

        end_fr1, start_fr2 = max(fr_rows(1, 26), default=-1), min(fr_rows(39, 55), default=None)
        end_fr2, start_fr3 = max(fr_rows(39, 55), default=None), min(fr_rows(66, 104), default=None)
        if start_fr2 is not None:
            number_segment("CDR1", end_fr1 + 1, start_fr2)
        if end_fr2 is not None and start_fr3 is not None:
            number_segment("CDR2", end_fr2 + 1, start_fr3)
        end_fr3 = max(fr_rows(66, 104), default=None)
        if end_fr3 is not None and end_fr3 + 1 < n:
            best = None
            for jr, jseq in self.j:
                if jr["locus"] != ref["locus"]:
                    continue
                for s0 in range(end_fr3 + 1, n):
                    seg = encode(query[s0:s0 + len(jseq)])
                    sc = SCORE[seg, encode(jseq)[: len(seg)]].sum() - (len(jseq) - len(seg)) * 0.5
                    if best is None or sc > best[0]:
                        best = (sc, s0, jr)
            _, s0, jr = best
            for o, qi in enumerate(range(s0, min(n, s0 + 11))):
                pos[qi] = str(118 + o)
            number_segment("CDR3", end_fr3 + 1, s0)
        return dict(positions=pos, v_allele=ref["allele"], v_gene=ref["gene"], species=ref["species"],
                    locus=ref["locus"], score=score)


def evaluate_aligner(aligner, pairs):
    """pairs: iterable of (example, domain). Returns a list of per-domain dicts."""
    rows = []
    for ex, d in pairs:
        q, truth = ex["seq"][d["start"]:d["end"]], ex["imgt_pos"][d["start"]:d["end"]]
        out = aligner.annotate(q)
        rows.append(dict(id=ex["id"], start=d["start"], n=len(q),
                         pos_correct=sum(a == b for a, b in zip(out["positions"], truth)),
                         gene_correct=(out["v_gene"] in d.get("gene_set", [d["v_gene"]])) and out["species"] == d["species"],
                         species_correct=out["species"] == d["species"], pred_positions=out["positions"]))
    return rows
