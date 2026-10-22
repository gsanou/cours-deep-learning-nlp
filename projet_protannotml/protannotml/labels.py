"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""Label spaces and example encoding for the ProtAnnotML multitask model (no torch dependency)."""
import json

from protannotml.imgt import pos_key, region_of, REGION_NAMES

IGNORE = -100
BIO_TAGS = ["O", "B-V", "I-V", "B-C", "I-C"]


def imgt_position_vocab(max_insertions=15):
    """All V-domain IMGT positions in chain order: 1..128 plus CDR3 insertions 111.1.. and ..112.1."""
    positions = [str(p) for p in range(1, 129)]
    positions += [f"111.{k}" for k in range(1, max_insertions + 1)] + [f"112.{k}" for k in range(1, max_insertions + 1)]
    return sorted(positions, key=pos_key)


class LabelSpace:
    def __init__(self, positions, loci, species, genes):
        self.positions, self.loci, self.species, self.genes = list(positions), list(loci), list(species), list(genes)
        self.pos_id = {p: i for i, p in enumerate(self.positions)}
        self.locus_id = {x: i for i, x in enumerate(self.loci)}
        self.species_id = {x: i for i, x in enumerate(self.species)}
        self.gene_id = {x: i for i, x in enumerate(self.genes)}

    @classmethod
    def from_examples(cls, examples):
        loci, species, genes = set(), set(), set()
        for ex in examples:
            for d in ex["domains"]:
                if d["type"] == "V":
                    loci.add(d["locus"]); species.add(d["species"])
                    genes.update(f"{d['species']}|{g}" for g in d.get("gene_set", [d["v_gene"]]))
        return cls(imgt_position_vocab(), sorted(loci), sorted(species), sorted(genes))

    def sizes(self):
        return dict(bio=len(BIO_TAGS), positions=len(self.positions), regions=len(REGION_NAMES),
                    loci=len(self.loci), species=len(self.species), genes=len(self.genes))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(positions=self.positions, loci=self.loci, species=self.species, genes=self.genes), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def encode_example(ex, space, max_residues=1022):
    """Per-residue integer labels (IGNORE where undefined) + per-V-domain targets. Truncates long sequences."""
    n = min(len(ex["seq"]), max_residues)
    bio = [BIO_TAGS.index(t) for t in ex["domain_tags"][:n]]
    pos = [space.pos_id.get(p, IGNORE) if p != "-" else IGNORE for p in ex["imgt_pos"][:n]]
    reg = [REGION_NAMES.index(r) if r in REGION_NAMES else IGNORE for r in ex["regions"][:n]]
    domains = []
    for d in ex["domains"]:
        if d["type"] != "V" or d["start"] >= n:
            continue
        genes = [space.gene_id[f"{d['species']}|{g}"] for g in d.get("gene_set", [d["v_gene"]]) if f"{d['species']}|{g}" in space.gene_id]
        domains.append(dict(start=d["start"], end=min(d["end"], n),
                            locus=space.locus_id.get(d["locus"], IGNORE),
                            species=space.species_id.get(d["species"], IGNORE),
                            genes=genes))                     # empty list = gene unseen in training: no gene loss
    return dict(id=ex["id"], seq=ex["seq"][:n], bio=bio, pos=pos, region=reg, domains=domains)
