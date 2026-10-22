"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""CPU smoke test of the whole ProtAnnotML stack with a tiny random ESM (no download needed).

Run from projet_protannotml/:  python scripts/smoke_test.py
"""
import itertools
import sys
import tempfile
from pathlib import Path

import torch
from transformers import EsmConfig, EsmTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protannotml.crf import ConstrainedCRF                       # noqa: E402
from protannotml.data import Collator                             # noqa: E402
from protannotml.inference import annotate                        # noqa: E402
from protannotml.labels import LabelSpace, encode_example         # noqa: E402
from protannotml.model import ProtAnnotModel                      # noqa: E402
from protannotml.synth import make_dataset, make_reference        # noqa: E402

ESM_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q",
             "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]


def tiny_esm():
    vocab = Path(tempfile.mkdtemp()) / "vocab.txt"
    vocab.write_text("\n".join(ESM_VOCAB))
    tok = EsmTokenizer(vocab_file=str(vocab))
    cfg = EsmConfig(vocab_size=len(ESM_VOCAB), pad_token_id=1, mask_token_id=32, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=4, intermediate_size=128, max_position_embeddings=1026,
                    position_embedding_type="rotary", token_dropout=True, emb_layer_norm_before=False)
    return tok, cfg


def check_crf():
    torch.manual_seed(0)
    crf = ConstrainedCRF(3)
    em = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    paths = crf.decode(em, mask)
    trans = crf.transitions + crf.trans_mask
    for b in range(2):
        n = int(mask[b].sum())
        def score(p):
            s = crf.start[p[0]] + em[b, 0, p[0]] + crf.end[p[-1]]
            return s + sum(trans[p[t - 1], p[t]] + em[b, t, p[t]] for t in range(1, n))
        brute = max(itertools.product(range(3), repeat=n), key=lambda p: float(score(p)))
        assert list(brute) == paths[b], (brute, paths[b])
        # NLL of the best path must be finite and positive
    nll = crf.neg_log_likelihood(em, torch.tensor([paths[0], paths[1] + [0]]), mask)
    assert torch.isfinite(nll) and nll > 0
    print("CRF: Viterbi == brute force, NLL finite")


def main():
    check_crf()
    tok, cfg = tiny_esm()
    ids = tok("ACDKW")["input_ids"]
    assert len(ids) == 7 and ids[0] == tok.cls_token_id and ids[-1] == tok.eos_token_id, ids
    print("tokenizer: one token per residue + <cls>/<eos>")

    refs, js, cs = make_reference(seed=0)
    data = make_dataset(refs, js, cs, n=64, seed=1)
    space = LabelSpace.from_examples(data)
    enc = [encode_example(e, space) for e in data]
    model = ProtAnnotModel(space.sizes(), config=cfg)
    collate = Collator(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(40):
        batch = collate(enc[(step * 8) % 64:(step * 8) % 64 + 8])
        res = model(**batch)
        assert torch.isfinite(res["loss"]), res["losses"]
        opt.zero_grad(); res["loss"].backward(); opt.step()
        losses.append(float(res["loss"]))
    print(f"loss: first 5 steps {sum(losses[:5]) / 5:.3f} -> last 5 steps {sum(losses[-5:]) / 5:.3f}")
    assert sum(losses[-5:]) < sum(losses[:5]), "loss did not decrease"
    out = annotate(model, tok, [data[0]["seq"], data[1]["seq"]], space)
    assert len(out) == 2 and all("domains" in o for o in out)
    print("inference OK:", [(d["type"], d["start"], d["end"]) for d in out[0]["domains"]])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
