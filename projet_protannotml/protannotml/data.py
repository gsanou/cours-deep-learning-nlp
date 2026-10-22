"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""Torch dataset and collate function (tokenizer-aware) for the multitask model."""
import torch
from torch.utils.data import Dataset

from protannotml.labels import IGNORE


class EncodedDataset(Dataset):
    def __init__(self, encoded):
        self.items = encoded

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        seqs = [b["seq"] for b in batch]
        enc = self.tok(seqs, padding=True, return_tensors="pt", add_special_tokens=True)
        L = max(len(s) for s in seqs)
        n_tokens = enc["input_ids"].size(1)
        assert n_tokens >= L + 2, "tokenizer must produce one token per residue (+ <cls>, <eos>)"
        residue_mask = torch.zeros(len(batch), L, dtype=torch.bool)
        labels = {k: torch.full((len(batch), L), IGNORE, dtype=torch.long) for k in ("bio", "pos", "region")}
        domains = []
        for i, b in enumerate(batch):
            n = len(b["seq"])
            residue_mask[i, :n] = True
            for k in labels:
                labels[k][i, :n] = torch.tensor(b[k], dtype=torch.long)
            for d in b["domains"]:
                domains.append(dict(batch=i, **d))
        return dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], residue_mask=residue_mask,
                    bio=labels["bio"], pos=labels["pos"], region=labels["region"], domains=domains)
