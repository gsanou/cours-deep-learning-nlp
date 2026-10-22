"""Exported from notebook pf06_echelle_deploiement.ipynb — edit the notebook source (src/pf06_echelle_deploiement.md), not this file."""

"""Regenerate the synthetic development data (notebooks pf01-pf02) into data_protannotml/ of the project.

Usage (from projet_protannotml/): python scripts/make_synthetic_data.py [--n 6000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protannotml.imgt import allele_key                                                     # noqa: E402
from protannotml.splits import cluster_at, encode_aligned, group_split, identity_matrix      # noqa: E402
from protannotml.synth import equivalence_sets, make_dataset, make_reference, save_jsonl, check_example  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data_protannotml")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(exist_ok=True)
    refs, js, cs = make_reference(seed=0)
    data = make_dataset(refs, js, cs, n=args.n, seed=1)
    assert all(check_example(e) for e in data)
    eq_a, eq_g = equivalence_sets(refs, "allele"), equivalence_sets(refs, "gene")
    for e in data:
        for d in e["domains"]:
            if d["type"] == "V":
                k = allele_key(d["species"], d["v_allele"])
                d["allele_set"], d["gene_set"] = sorted(eq_a[k]), sorted(eq_g[k])
    keys = [allele_key(r["species"], r["allele"]) for r in refs]
    _, labels = cluster_at(identity_matrix(encode_aligned([r["positions"] for r in refs])), 0.80)
    cluster_of = dict(zip(keys, labels))
    rng = np.random.default_rng(0)
    ids = [e["id"] for e in data]; rng.shuffle(ids); n = len(ids)
    splits = {"R0 aléatoire": {"train": ids[: int(0.8 * n)], "val": ids[int(0.8 * n): int(0.9 * n)], "test": ids[int(0.9 * n):], "dropped": []},
              "R1 par allèle": group_split(data, lambda d: allele_key(d["species"], d["v_allele"]), seed=0),
              "R2 cluster 80 %": group_split(data, lambda d: int(cluster_of[allele_key(d["species"], d["v_allele"])]), seed=0)}
    rest = [e["id"] for e in data if e["species"] != "Mus musculus"]; rng.shuffle(rest)
    splits["R3 espèce exclue"] = {"train": rest[: int(0.9 * len(rest))], "val": rest[int(0.9 * len(rest)):],
                                  "test": [e["id"] for e in data if e["species"] == "Mus musculus"], "dropped": []}
    save_jsonl(data, out / "examples.jsonl"); save_jsonl(refs, out / "reference_V.jsonl")
    save_jsonl(js, out / "reference_J.jsonl"); save_jsonl(cs, out / "reference_C.jsonl")
    (out / "splits.json").write_text(json.dumps(splits))
    print(f"wrote {len(data)} examples and {len(splits)} split regimes to {out}/")


if __name__ == "__main__":
    main()
