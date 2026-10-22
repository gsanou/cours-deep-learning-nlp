"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""From sequences to DomainGapAlign-like annotations."""
import torch
import torch.nn.functional as F

from protannotml.decode import monotone_viterbi, spans_from_bio
from protannotml.imgt import region_of
from protannotml.labels import BIO_TAGS


@torch.no_grad()
def annotate(model, tokenizer, seqs, space, device="cpu", gene_threshold=0.5, top_k=3):
    model.eval()
    enc = tokenizer(seqs, padding=True, return_tensors="pt")
    L = max(len(s) for s in seqs)
    residue_mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        residue_mask[i, :len(s)] = True
    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), residue_mask.to(device))
    bio_paths = model.crf.decode(out["bio_logits"], residue_mask.to(device))
    pos_logp = F.log_softmax(out["pos_logits"].float(), dim=-1).cpu().numpy()
    results = []
    for i, s in enumerate(seqs):
        tags = [BIO_TAGS[t] for t in bio_paths[i]]
        domains = []
        for dtype, start, end in spans_from_bio(tags):
            dom = dict(type=dtype, start=start, end=end, sequence=s[start:end])
            if dtype == "V" and end - start > len(space.positions):
                dom["warning"] = "predicted V span longer than the IMGT position space: domain boundary is probably wrong"
            elif dtype == "V":
                path = monotone_viterbi(pos_logp[i, start:end])
                positions = [space.positions[k] for k in path]
                dom["imgt_positions"] = positions
                dom["regions"] = [region_of(p) for p in positions]
                z = model.pool_domains(out["states"], [(i, start, end)])
                p_locus = F.softmax(model.locus_head(z).float(), -1)[0]
                p_species = F.softmax(model.species_head(z).float(), -1)[0]
                p_gene = F.softmax(model.gene_head(z).float(), -1)[0]
                top = torch.topk(p_gene, k=min(top_k, p_gene.numel()))
                dom["locus"] = space.loci[int(p_locus.argmax())]
                dom["species"] = space.species[int(p_species.argmax())]
                dom["gene_candidates"] = [(space.genes[int(k)], round(float(p), 3)) for p, k in zip(top.values, top.indices)]
                dom["gene"] = dom["gene_candidates"][0][0] if float(top.values[0]) >= gene_threshold else None   # abstain
            domains.append(dom)
        results.append(dict(sequence=s, domains=domains))
    return results
