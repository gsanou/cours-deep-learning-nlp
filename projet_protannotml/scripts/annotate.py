"""Exported from notebook pf06_echelle_deploiement.ipynb — edit the notebook source (src/pf06_echelle_deploiement.md), not this file."""

"""Annotate a FASTA file with a trained ProtAnnotML run; writes an AIRR-like TSV (one row per domain).

Usage: python scripts/annotate.py --run runs/h100 --fasta input.fasta --out annotations.tsv [--device cuda]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protannotml.inference import annotate                     # noqa: E402
from protannotml.labels import LabelSpace                       # noqa: E402
from protannotml.model import ProtAnnotModel                    # noqa: E402

REGION_COLUMNS = [("FR1", "fwr1_aa"), ("CDR1", "cdr1_aa"), ("FR2", "fwr2_aa"), ("CDR2", "cdr2_aa"),
                  ("FR3", "fwr3_aa"), ("CDR3", "cdr3_aa"), ("FR4", "fwr4_aa")]


def read_fasta(path):
    name, chunks = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(chunks)
            name, chunks = line[1:].split()[0], []
        elif line:
            chunks.append(line.upper())
    if name is not None:
        yield name, "".join(chunks)


def load_run(run_dir, device):
    run = Path(run_dir)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    space = LabelSpace.load(run / "label_space.json")
    model = ProtAnnotModel(space.sizes(), backbone=cfg["backbone"])
    if cfg.get("lora", False):
        from peft import LoraConfig, get_peft_model
        model.backbone = get_peft_model(model.backbone, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                                                   target_modules=["query", "key", "value", "dense"]))
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), AutoTokenizer.from_pretrained(cfg["backbone"]), space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--fasta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--gene_threshold", type=float, default=0.5)
    args = ap.parse_args()
    model, tok, space = load_run(args.run, args.device)
    records = sorted(read_fasta(args.fasta), key=lambda r: len(r[1]))         # length-sorted batches: less padding
    fields = ["sequence_id", "domain_index", "domain_type", "start", "end", "locus", "species", "v_call",
              "v_call_candidates"] + [c for _, c in REGION_COLUMNS] + ["imgt_numbering"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for k in range(0, len(records), args.batch):
            chunk = records[k:k + args.batch]
            results = annotate(model, tok, [s for _, s in chunk], space, device=args.device, gene_threshold=args.gene_threshold)
            for (name, _), res in zip(chunk, results):
                for i, d in enumerate(res["domains"]):
                    row = dict(sequence_id=name, domain_index=i, domain_type=d["type"], start=d["start"] + 1, end=d["end"])
                    if d["type"] == "V":
                        regions = {r: "" for r, _ in REGION_COLUMNS}
                        for aa, r in zip(d["sequence"], d["regions"]):
                            regions[r] += aa
                        row.update(locus=d["locus"], species=d["species"], v_call=(d["gene"] or "").split("|")[-1],
                                   v_call_candidates=json.dumps(d["gene_candidates"]),
                                   imgt_numbering=json.dumps(list(zip(d["imgt_positions"], d["sequence"]))),
                                   **{col: regions[r] for r, col in REGION_COLUMNS})
                    w.writerow(row)
    print(f"annotated {len(records)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
