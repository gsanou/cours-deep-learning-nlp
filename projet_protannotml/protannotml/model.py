"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""ProtAnnotML multitask model: PLM backbone + domain CRF + IMGT position / region heads + per-domain heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel

from protannotml.crf import ConstrainedCRF
from protannotml.decode import bio_mask
from protannotml.labels import IGNORE


class ProtAnnotModel(nn.Module):
    TASKS = ["bio", "positions", "regions", "locus", "species", "gene"]

    def __init__(self, sizes, backbone="facebook/esm2_t12_35M_UR50D", config=None, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.backbone = EsmModel(config, add_pooling_layer=False) if config is not None else \
            EsmModel.from_pretrained(backbone, add_pooling_layer=False)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        H = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.bio_head = nn.Linear(H, sizes["bio"])
        self.pos_head = nn.Linear(H, sizes["positions"])
        self.region_head = nn.Linear(H, sizes["regions"])
        self.domain_mlp = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout))
        self.locus_head = nn.Linear(H, sizes["loci"])
        self.species_head = nn.Linear(H, sizes["species"])
        self.gene_head = nn.Linear(H, sizes["genes"])
        allowed, start = bio_mask()
        self.crf = ConstrainedCRF(sizes["bio"], allowed, start)
        self.log_vars = nn.Parameter(torch.zeros(len(self.TASKS)))          # homoscedastic uncertainty weighting

    def residue_states(self, input_ids, attention_mask, n_residues):
        """Hidden states aligned with residues: token 0 is <cls>, residues are tokens 1..L."""
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.dropout(h[:, 1:1 + n_residues])

    def pool_domains(self, h, spans):
        """spans: list of (batch_index, start, end). Mean of residue states over each domain."""
        if not spans:
            return h.new_zeros(0, h.size(-1))
        pooled = torch.stack([h[b, s:e].mean(0) for b, s, e in spans])
        return self.domain_mlp(pooled)

    def forward(self, input_ids, attention_mask, residue_mask, bio=None, pos=None, region=None, domains=None):
        L = residue_mask.size(1)
        h = self.residue_states(input_ids, attention_mask, L)
        out = dict(bio_logits=self.bio_head(h), pos_logits=self.pos_head(h), region_logits=self.region_head(h), states=h)
        if bio is None:
            return out
        losses = {}
        bio_tags = bio.clamp(min=0)                                          # padding -> any valid tag (masked)
        losses["bio"] = self.crf.neg_log_likelihood(out["bio_logits"], bio_tags, residue_mask)
        losses["positions"] = F.cross_entropy(out["pos_logits"].flatten(0, 1).float(), pos.flatten(), ignore_index=IGNORE)
        losses["regions"] = F.cross_entropy(out["region_logits"].flatten(0, 1).float(), region.flatten(), ignore_index=IGNORE)
        spans = [(d["batch"], d["start"], d["end"]) for d in domains]
        z = self.pool_domains(h, spans)
        # zero that touches the backbone and every domain head: keeps all parameters in the graph (DDP needs it)
        zero = h.sum() * 0.0 + sum(p.sum() for m in (self.domain_mlp, self.locus_head, self.species_head, self.gene_head)
                                   for p in m.parameters()) * 0.0
        if len(spans):
            locus = torch.tensor([d["locus"] for d in domains], device=h.device)
            species = torch.tensor([d["species"] for d in domains], device=h.device)
            losses["locus"] = F.cross_entropy(self.locus_head(z).float(), locus, ignore_index=IGNORE)
            losses["species"] = F.cross_entropy(self.species_head(z).float(), species, ignore_index=IGNORE)
            gene_logp = F.log_softmax(self.gene_head(z).float(), dim=-1)
            gene_terms = [-gene_logp[k, d["genes"]].mean() for k, d in enumerate(domains) if d["genes"]]
            losses["gene"] = torch.stack(gene_terms).mean() if gene_terms else zero
        else:
            losses["locus"] = losses["species"] = losses["gene"] = zero
        for k in ("positions", "regions", "locus", "species"):
            if torch.isnan(losses[k]):                                       # batch without any valid label
                losses[k] = zero
        total = zero
        for i, task in enumerate(self.TASKS):
            total = total + torch.exp(-self.log_vars[i]) * losses[task] + self.log_vars[i]
        out["loss"], out["losses"] = total, {k: float(v.detach()) for k, v in losses.items()}
        return out
