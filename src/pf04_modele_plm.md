# Fil rouge ProtAnnotML · 4 — Le modèle : PLM multi-tâche, CRF et décodage contraint

> **Objectif** : écrire le modèle complet de ProtAnnotML et ses scripts d'entraînement : un **PLM** (ESM-2 par défaut) en tronc commun, des **têtes** par tâche (domaines avec CRF contraint, positions IMGT, régions, locus/espèce/gène au niveau du domaine avec cibles « ensemblistes »), une **pondération des pertes** par incertitude, l'adaptation au domaine (**DAPT** par MLM) et l'affinage (complet ou **LoRA**). Un **test rapide** sur CPU avec un mini-modèle aléatoire permet de valider toute la chaîne avant de réserver des GPU.

| | |
|---|---|
| **Durée estimée** | 3 h |
| **Prérequis** | Fil rouge 1–3 ; modules 7–10 (Hugging Face, LoRA, PLM, CRF), 13 (multi-tâche) |
| **Librairies** | `torch`, `transformers`, `peft`, `datasets`, `pyyaml` (sections 3–7) ; `numpy` (sections 1–2) |

> ⚠️ **Ce qui a été exécuté.** Les sections 1–2 (espaces d'étiquettes, encodage des exemples, cibles ensemblistes) sont exécutées ici. Le code PyTorch (sections 3–7) n'a **pas pu être exécuté** dans mon environnement (ni PyTorch ni accès à PyPI) : il a été relu et vérifié statiquement, et le script `scripts/smoke_test.py` est fait pour que **toi**, tu le valides en deux minutes sur ton poste (CPU ou GPU de 6 Go) avant de lancer quoi que ce soit sur le V100 ou Jean Zay. Lance-le en premier ; s'il échoue, l'erreur sera dans une petite fonction, pas au bout de 10 h de calcul.

**Plan**

1. L'architecture en une figure
2. Espaces d'étiquettes et encodage des exemples
3. Le CRF contraint en PyTorch
4. Le modèle multi-tâche
5. Données, collation et alignement tokens / résidus
6. Entraînement : DAPT, affinage, LoRA, précision mixte, reprise
7. Inférence : de la séquence à l'annotation
8. Le test rapide (à lancer en premier)
9. Récapitulatif et exercices

```python
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.figsize": (7, 4), "axes.prop_cycle": plt.cycler(color=PALETTE),
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "lines.linewidth": 2, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
ROOT = Path.cwd().parent / "projet_protannotml"
sys.path.insert(0, str(ROOT))
DATA = Path("data_protannotml")
from protannotml.synth import load_jsonl

examples = load_jsonl(DATA / "examples.jsonl")
splits = json.loads((DATA / "splits.json").read_text())
```

+++

## 1. L'architecture en une figure

```
séquence  ──► tokenizer ESM ──► <cls> r1 r2 … rL <eos>
                                   │
                         PLM (ESM-2, 35M → 650M ; DAPT puis affinage complet ou LoRA)
                                   │  états cachés h1 … hL (dimension H)
         ┌─────────────────────────┼──────────────────────────┬──────────────────────────────┐
   tête domaines (5)         tête positions (158)       tête régions (7)          regroupement moyen par domaine V
   + CRF contraint BIO       + Viterbi monotone          (tâche auxiliaire)        ├─ locus (4)
   T1                        T3  (régions T2 déduites)                             ├─ espèce
                                                                                   └─ gène (cible = ensemble)  T4
   perte totale  L = Σ_k exp(−s_k) · L_k + s_k      (s_k = log-variance apprise, module 13)
```

Choix de conception, et pourquoi :
- **Un seul tronc partagé** : les quatre tâches reposent sur la même compréhension de la séquence (frameworks, ancres, CDR) ; c'est le cadre multi-tâche du module 13, avec la pondération par incertitude que tu as retenue pour ProtAnnotML.
- **Les régions sont déduites des positions** (une position IMGT appartient à une seule région) : cela garantit la cohérence T2/T3. La tête régions reste comme tâche auxiliaire d'entraînement (signal plus grossier, plus facile au début).
- **Le CRF ne sert qu'aux domaines** (5 étiquettes, transitions interdites codées en dur). Pour les positions (158 états), le Viterbi monotone du notebook 3 suffit au décodage, sans paramètres de transition.
- **T4 au niveau du domaine** : on regroupe les états cachés **sur le domaine V** (prédit à l'inférence, réel à l'entraînement), pas sur toute la séquence : un scFv a deux domaines, de loci différents.

+++

## 2. Espaces d'étiquettes et encodage des exemples

L'espace des positions est **fixe** (1–128, plus 111.1–111.15 et 112.15–112.1, dans l'ordre de la chaîne) : il ne dépend pas des données, ce qui permet de réutiliser un modèle sur un autre jeu. Les espaces locus / espèce / gène sont construits sur l'entraînement et **sauvegardés avec le modèle** (sinon, impossible de relire les prédictions plus tard).

Pour le gène, la cible n'est pas une classe mais un **ensemble** (les gènes à séquence protéique identique, notebook 1). On entraîne avec une **cible douce** uniforme sur l'ensemble : $L = -\sum_{g \in G} \frac{1}{|G|} \log p(g)$. Le modèle n'est pas pénalisé pour répartir sa masse entre des gènes réellement indiscernables.

```python export=protannotml/labels.py
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
```

```python
from protannotml.labels import LabelSpace, encode_example, imgt_position_vocab, BIO_TAGS, IGNORE

by_id = {ex["id"]: ex for ex in examples}
train = [by_id[i] for i in splits["R2 cluster 80 %"]["train"]]
space = LabelSpace.from_examples(train)
print("label space sizes:", space.sizes())
print("first/last positions:", space.positions[:3], "...", space.positions[108:114], "...", space.positions[-3:])

enc = encode_example(next(ex for ex in train if ex["kind"] == "scFv"), space)
print(enc["id"], "length", len(enc["seq"]), "| domains:", [(d["start"], d["end"], space.loci[d["locus"]], len(d["genes"])) for d in enc["domains"]])
print("residues with a position label:", sum(p != IGNORE for p in enc["pos"]), "/", len(enc["pos"]))

test = [by_id[i] for i in splits["R2 cluster 80 %"]["test"]]
unseen = sum(1 for ex in test for d in encode_example(ex, space)["domains"] if not d["genes"])
total = sum(1 for ex in test for d in ex["domains"] if d["type"] == "V")
print(f"R2 test V domains whose gene is absent from the training label space: {unseen}/{total}")
```

Le dernier chiffre rappelle la conséquence du régime R2 (notebook 2) : en test, **tous** les gènes sont hors de l'espace d'étiquettes. Le classifieur de gènes ne peut qu'**errer ou s'abstenir** ; on mesurera donc, en R2, la qualité de l'abstention, et l'exactitude du gène en R0/R1.

+++

## 3. Le CRF contraint en PyTorch

Le CRF linéaire du module 10, vectorisé par lot, avec deux ajouts : les transitions interdites (BIO) reçoivent un score de −10⁴ (fixe, non appris), et les calculs se font en `float32` même en précision mixte (les `logsumexp` en `bfloat16` perdent trop de précision).

```python noexec export=protannotml/crf.py
"""Linear-chain CRF with hard transition constraints, batched (forward algorithm + Viterbi)."""
import torch
import torch.nn as nn

FORBIDDEN = -1e4


class ConstrainedCRF(nn.Module):
    def __init__(self, n_tags, allowed=None, start_allowed=None):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        allowed = torch.ones(n_tags, n_tags, dtype=torch.bool) if allowed is None else torch.as_tensor(allowed, dtype=torch.bool)
        start_allowed = torch.ones(n_tags, dtype=torch.bool) if start_allowed is None else torch.as_tensor(start_allowed, dtype=torch.bool)
        self.register_buffer("trans_mask", torch.where(allowed, 0.0, FORBIDDEN))
        self.register_buffer("start_mask", torch.where(start_allowed, 0.0, FORBIDDEN))

    def _trans(self):
        return self.transitions + self.trans_mask

    def neg_log_likelihood(self, emissions, tags, mask):
        """emissions (B, T, K), tags (B, T) long (any valid id on padding), mask (B, T) bool with mask[:, 0] all True."""
        emissions = emissions.float()
        trans, start = self._trans(), self.start + self.start_mask
        B, T, K = emissions.shape
        # score of the gold path
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            step = trans[tags[:, t - 1], tags[:, t]] + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end[last]
        # log partition function (forward algorithm)
        alpha = start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        log_z = torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)
        return (log_z - score).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Batched Viterbi. Returns a list of tag-id lists (one per sequence, unpadded)."""
        emissions = emissions.float()
        trans, start = self._trans(), self.start + self.start_mask
        B, T, K = emissions.shape
        delta = start.unsqueeze(0) + emissions[:, 0]
        back = []
        for t in range(1, T):
            scores = delta.unsqueeze(2) + trans.unsqueeze(0)              # (B, K_prev, K)
            best, arg = scores.max(dim=1)
            nxt = best + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            delta = torch.where(m, nxt, delta)
            back.append(torch.where(m, arg, torch.arange(K, device=arg.device).expand(B, K)))  # identity on padding
        delta = delta + self.end.unsqueeze(0)
        best_last = delta.argmax(1)
        paths = []
        lengths = mask.sum(1).tolist()
        for b in range(B):
            tag = int(best_last[b])
            path = [tag]
            for t in range(T - 2, -1, -1):
                tag = int(back[t][b, tag])
                path.append(tag)
            path = path[::-1]
            paths.append(path[: lengths[b]])
        return paths
```

> Sur le rembobinage : pour les positions de remplissage, le pointeur arrière est l'**identité** (on reste sur la même étiquette), donc remonter depuis la fin du tableau traverse le remplissage sans rien changer, et l'on tronque à la longueur réelle. C'est le détail qui casse le plus souvent les implémentations de CRF par lot ; le test rapide le vérifie contre une recherche exhaustive.

+++

## 4. Le modèle multi-tâche

```python noexec export=protannotml/model.py
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
```

Deux points de vigilance :
- `F.cross_entropy` avec `ignore_index` renvoie **NaN** si **aucune** étiquette du lot n'est valide (division par zéro) : d'où la vérification explicite. Un seul NaN détruit tout l'entraînement en quelques pas.
- Le terme `zero` (une somme nulle qui touche le tronc et toutes les têtes de domaine) maintient tous les paramètres dans le graphe de calcul : sans lui, un lot sans domaine V laisse des paramètres sans gradient, et l'entraînement distribué (DDP) s'arrête avec une erreur de réduction incomplète.

+++

## 5. Données, collation et alignement tokens / résidus

Le tokenizer d'ESM découpe une chaîne d'acides aminés en un token par résidu et ajoute `<cls>` au début et `<eos>` à la fin : le résidu $i$ est le token $i+1$. Le modèle prend les états `h[:, 1:1+L]`, donc toutes les étiquettes sont alignées sur les **résidus**, avec un masque de résidus pour le remplissage.

```python noexec export=protannotml/data.py
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
```

+++

## 6. Entraînement : DAPT, affinage, LoRA, précision mixte, reprise

### 6.1 Adaptation au domaine (DAPT) par MLM

On poursuit le pré-entraînement d'ESM-2 par **modèle de langage masqué** sur des séquences d'immunoglobulines et de récepteurs T (IMGT/LIGM-DB, éventuellement OAS), dédupliquées et regroupées par identité pour ne pas sur-représenter les clonotypes abondants. Le script est volontairement standard (Hugging Face `Trainer`), car il n'a rien de spécifique.

```python noexec export=scripts/dapt_mlm.py
"""Domain-adaptive MLM pretraining of an ESM-2 checkpoint on IG/TR sequences (one sequence per line).

Usage: python scripts/dapt_mlm.py --model facebook/esm2_t12_35M_UR50D --train ig_train.txt --valid ig_valid.txt --out runs/dapt35M
"""
import argparse
import os

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, EsmForMaskedLM, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=256)              # V domains and Fv chains are short
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_steps", type=int, default=50_000)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model)
    ds = load_dataset("text", data_files={"train": args.train, "validation": args.valid})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=args.mlm_probability)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()       # H100 / A100: bf16 ; V100: fp16
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, warmup_ratio=0.05, weight_decay=0.01,
        max_steps=args.max_steps, lr_scheduler_type="cosine", bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True, eval_strategy="steps", eval_steps=2000, save_steps=2000, save_total_limit=3,
        logging_steps=100, dataloader_num_workers=min(8, os.cpu_count() or 1), group_by_length=True,
        report_to="none", seed=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"], data_collator=collator)
    last = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    trainer.train(resume_from_checkpoint=last)                       # resumes automatically after a Slurm time limit
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
```

> 🧪 **Biais de germline.** Si le corpus de DAPT est dominé par des répertoires naïfs, le modèle apprend à « prédire la germline » (c'est le constat d'AbLang-2 : les modèles d'anticorps prédisaient la germline dans environ 85 % des cas pour des résidus non germinaux masqués). Pour ProtAnnotML, c'est plutôt un **atout** pour T3/T4 (reconnaître la germline sous les mutations), mais un risque si tu veux plus tard des représentations utiles pour la maturation. Garde une trace de la composition du corpus et évalue la perplexité **séparément** sur les résidus germinaux et mutés.

### 6.2 Affinage multi-tâche

Le script d'affinage est une boucle PyTorch explicite (plus transparente qu'un `Trainer` pour un modèle multi-tâche) qui gère : la configuration YAML par matériel, l'entraînement distribué (`torchrun`), la précision mixte adaptée au GPU (bf16 sur H100, fp16 + *loss scaling* sur V100), le *gradient checkpointing*, LoRA optionnel, deux taux d'apprentissage (tronc / têtes), et la **reprise** après interruption (indispensable avec la limite de 20 h des files Jean Zay).

```python noexec export=scripts/train_multitask.py
"""Multitask fine-tuning of ProtAnnotML.

Single GPU : python scripts/train_multitask.py --config configs/gpu6gb.yaml
Multi-GPU  : torchrun --nproc_per_node=4 scripts/train_multitask.py --config configs/h100.yaml
Slurm      : srun python scripts/train_multitask.py --config configs/h100.yaml   (one task per GPU, see slurm/)
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protannotml.data import Collator, EncodedDataset          # noqa: E402
from protannotml.labels import LabelSpace, encode_example        # noqa: E402
from protannotml.model import ProtAnnotModel                     # noqa: E402
from protannotml.synth import load_jsonl                         # noqa: E402

STOP = {"flag": False}


def on_signal(signum, frame):                                    # Slurm sends SIGUSR1 before the time limit (--signal)
    STOP["flag"] = True


def setup_distributed():
    env = os.environ
    if "WORLD_SIZE" not in env and int(env.get("SLURM_NTASKS", "1")) > 1:   # launched by srun, one task per GPU
        env["WORLD_SIZE"], env["RANK"], env["LOCAL_RANK"] = env["SLURM_NTASKS"], env["SLURM_PROCID"], env["SLURM_LOCALID"]
    if "WORLD_SIZE" in env and int(env["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return rank, dist.get_world_size(), torch.device("cuda", local)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def apply_lora(model, cfg):
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg.get("lora_dropout", 0.05),
                      target_modules=["query", "key", "value", "dense"], bias="none")
    model.backbone = get_peft_model(model.backbone, lcfg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value pairs, e.g. lr_backbone=1e-5")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kv in args.override:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    rank, world, device = setup_distributed()
    set_seed(cfg.get("seed", 0) + rank)
    signal.signal(signal.SIGUSR1, on_signal)
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    examples = {e["id"]: e for e in load_jsonl(cfg["examples"])}
    split = json.load(open(cfg["splits"]))[cfg["regime"]]
    train_ex, val_ex = [examples[i] for i in split["train"]], [examples[i] for i in split["val"]]
    space = LabelSpace.from_examples(train_ex)
    if rank == 0:
        space.save(out / "label_space.json")
        yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    train_ds = EncodedDataset([encode_example(e, space, cfg["max_residues"]) for e in train_ex])
    val_ds = EncodedDataset([encode_example(e, space, cfg["max_residues"]) for e in val_ex])
    sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    collate = Collator(tok)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, shuffle=sampler is None,
                          collate_fn=collate, num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate, num_workers=2)

    model = ProtAnnotModel(space.sizes(), backbone=cfg["backbone"], gradient_checkpointing=cfg.get("gradient_checkpointing", False))
    if cfg.get("lora", False):
        model = apply_lora(model, cfg)
    model.to(device)
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": cfg["lr_backbone"]},
                             {"params": head_params, "lr": cfg["lr_heads"]}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_dl) / cfg["grad_accum"])
    total_steps, warmup = steps_per_epoch * cfg["epochs"], int(0.06 * steps_per_epoch * cfg["epochs"])
    def lr_lambda(s):                                                       # linear warmup, then linear decay
        if s < warmup:
            return (s + 1) / max(1, warmup)
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    ddp_model = DDP(model, device_ids=[device.index]) if world > 1 else model

    start_epoch, step = 0, 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():                                                   # resume
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start_epoch, step = ck["epoch"], ck["step"]
        if rank == 0:
            print(f"resumed from epoch {start_epoch}, step {step}", flush=True)

    def save(epoch):
        if rank == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, step=step), out / "last.tmp")
            os.replace(out / "last.tmp", ckpt_path)                         # atomic: never a half-written checkpoint

    def should_stop():
        """Collective decision: every rank must leave the loop at the same step, otherwise DDP hangs."""
        flag = torch.tensor([1.0 if STOP["flag"] else 0.0], device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def to_device(batch):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    log = open(out / "log.jsonl", "a") if rank == 0 else None
    for epoch in range(start_epoch, cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp_model.train()
        t0 = time.time()
        for it, batch in enumerate(train_dl):
            batch = to_device(batch)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                res = ddp_model(**batch)
                loss = res["loss"] / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (it + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if rank == 0 and step % cfg.get("log_every", 50) == 0:
                    log.write(json.dumps(dict(epoch=epoch, step=step, loss=float(res["loss"]), **res["losses"],
                                              weights=torch.exp(-model.log_vars.detach()).tolist())) + "\n"); log.flush()
                if should_stop():
                    STOP["flag"] = True
                    break
        save(epoch + (0 if STOP["flag"] else 1))
        if world > 1:
            dist.barrier()                                                   # nobody continues before the checkpoint exists
        if STOP["flag"]:
            if rank == 0:
                print("stopping on signal; checkpoint saved, resubmit the job to resume", flush=True)
            break
        # quick validation: token-level accuracies (full decoding: protannotml.inference.annotate, scripts/annotate.py)
        model.eval(); correct = total = bio_ok = bio_tot = 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            for batch in val_dl:
                batch = to_device(batch)
                res = model(batch["input_ids"], batch["attention_mask"], batch["residue_mask"])
                m = batch["pos"] != -100
                correct += (res["pos_logits"].argmax(-1)[m] == batch["pos"][m]).sum().item(); total += m.sum().item()
                rm = batch["residue_mask"]
                bio_ok += (res["bio_logits"].argmax(-1)[rm] == batch["bio"][rm]).sum().item(); bio_tot += rm.sum().item()
        if rank == 0:
            log.write(json.dumps(dict(epoch=epoch, val_pos_acc=correct / max(1, total), val_bio_acc=bio_ok / max(1, bio_tot),
                                      epoch_minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
            print(f"epoch {epoch}: val position acc {correct / max(1, total):.4f}, bio acc {bio_ok / max(1, bio_tot):.4f}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Les configurations par matériel (détaillées au notebook 6) :

```python exec
CONFIGS = {
    "gpu6gb": dict(backbone="facebook/esm2_t12_35M_UR50D", batch_size=8, grad_accum=4, gradient_checkpointing=True,
                   lora=False, lr_backbone=3e-5, lr_heads=1e-3, epochs=10),
    "v100": dict(backbone="facebook/esm2_t33_650M_UR50D", batch_size=8, grad_accum=4, gradient_checkpointing=True,
                 lora=True, lora_r=16, lora_alpha=32, lr_backbone=2e-4, lr_heads=1e-3, epochs=8),
    "h100": dict(backbone="facebook/esm2_t33_650M_UR50D", batch_size=16, grad_accum=1, gradient_checkpointing=False,
                 lora=False, lr_backbone=2e-5, lr_heads=1e-3, epochs=8),
}
common = dict(examples="data_protannotml/examples.jsonl", splits="data_protannotml/splits.json", regime="R2 cluster 80 %",
              max_residues=1022, num_workers=4, log_every=50, seed=0)
import yaml
(ROOT / "configs").mkdir(exist_ok=True)
for name, c in CONFIGS.items():
    full = dict(common, **c, output_dir=f"runs/{name}")
    (ROOT / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(full, sort_keys=False))
print(sorted(p.name for p in (ROOT / "configs").iterdir()))
print((ROOT / "configs" / "v100.yaml").read_text())
```

> Avec LoRA, le taux d'apprentissage du tronc s'applique aux **adaptateurs** : il est donc plus élevé (2·10⁻⁴) qu'en affinage complet (2–3·10⁻⁵). Sur les têtes, 10⁻³ est un point de départ raisonnable ; c'est le premier hyperparamètre à balayer.

+++

## 7. Inférence : de la séquence à l'annotation

L'inférence enchaîne : CRF des domaines → pour chaque domaine V prédit, Viterbi monotone sur les positions (restreint au domaine) → régions déduites → regroupement du domaine et têtes locus / espèce / gène, avec **abstention** si la probabilité maximale du gène est sous un seuil (à calibrer sur la validation, notebook 5).

```python noexec export=protannotml/inference.py
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
```

+++

## 8. Le test rapide (à lancer en premier)

Ce script construit un **mini-ESM aléatoire** (2 couches, dimension 64) et son tokenizer à partir d'un vocabulaire écrit localement : **aucun téléchargement**, donc il tourne aussi sur un nœud de calcul sans Internet. Il vérifie : le CRF contre une recherche exhaustive, l'alignement tokens / résidus, une décroissance de la perte sur 40 pas, l'absence de NaN, et l'inférence de bout en bout. Compte une à deux minutes sur CPU.

```python noexec export=scripts/smoke_test.py
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
```

Ce que tu dois voir : `CRF: Viterbi == brute force`, une perte qui décroît, et `SMOKE TEST PASSED`. Les annotations du mini-modèle aléatoire n'ont évidemment aucun sens : on teste la **plomberie**, pas la qualité.

+++

## 9. Récapitulatif

| Élément | Choix |
|---|---|
| Tronc | ESM-2 (35M sur 6 Go, 650M sur V100/H100) ; AMPLIFY possible avec un adaptateur |
| T1 | tête BIO + CRF à transitions interdites (calcul en float32) |
| T3 / T2 | tête positions (158) + Viterbi monotone ; régions déduites ; tête régions auxiliaire |
| T4 | regroupement par domaine V ; cible douce sur les ensembles de gènes ; abstention par seuil |
| Pertes | pondération par incertitude (log-variances apprises) |
| Robustesse | NaN si aucune étiquette valide ; graphe connecté pour DDP ; checkpoint atomique ; reprise sur signal |
| Validation | `scripts/smoke_test.py` avant tout calcul coûteux |

### 🏋️ Exercices

1. **Lance le test rapide** sur ton poste, puis un vrai entraînement court (`gpu6gb.yaml`, 1 époque) sur les données synthétiques. La précision de position en validation dépasse-t-elle celle de la régression logistique du notebook 3 ?
2. **Ablations.** Entraîne sans la tête régions, puis sans pondération par incertitude (poids égaux). Quel est l'effet sur T1 et T3 ? (Rapporte la variance sur 3 graines avant de conclure.)
3. **AMPLIFY.** Écris un adaptateur qui charge `chandar-lab/AMPLIFY_120M` (`trust_remote_code=True`) et expose les états cachés comme `EsmModel` ; vérifie l'alignement tokens / résidus de son tokenizer avant toute chose.
4. **Allèles.** Ajoute une tête allèle, ou mieux : prédis le gène, puis choisis l'allèle par alignement sur les seuls allèles de ce gène (hybride modèle + alignement). Lequel est le plus juste ? Le plus rapide ?

<details><summary>Indications</summary>

2. Les ablations en multi-tâche sont bruitées : 3 graines minimum, et un test apparié sur les mêmes séquences de test (notebook 5).
3. Le tokenizer d'AMPLIFY peut ne pas ajouter les mêmes jetons spéciaux qu'ESM : c'est exactement l'assertion de `Collator` qui doit t'alerter.
4. L'hybride est souvent le meilleur compromis : le modèle réduit l'espace de recherche (rapide, robuste), l'alignement donne une réponse exacte et vérifiable au niveau allèle.
</details>
