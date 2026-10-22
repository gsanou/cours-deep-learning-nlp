# ProtAnnotML — model card (template)

## Intended use
Annotation of immunoglobulin and T cell receptor protein sequences: domain delimitation (V, C), IMGT unique
numbering of V domains, FR-IMGT / CDR-IMGT delimitation, locus / species / V gene identification with abstention.
Not intended for: non-IG/TR proteins, clinical decisions, sequences outside the training species without checking.

## Training data
IMGT release: <version, date>. Sources: <IMGT/LIGM-DB, IMGT/HighV-QUEST outputs, IMGT/mAb-DB>, filters: <...>.
Number of sequences per locus and species: <table>. Known biases: <naive repertoires over-represented, species coverage>.

## Evaluation
Regimes: homology-aware split (germline clusters at <x>% identity), held-out species <...>, temporal split on IMGT/mAb-DB.
Metrics (95% cluster-bootstrap CI): per-residue and per-domain numbering accuracy, CDR boundary errors,
gene identification with risk-coverage curve. Reference method: IMGT/DomainGapAlign on the same test sequences.

## Limitations
- Gene/allele calls are only defined for genes present in training; below the confidence threshold the model abstains.
- Unusual formats (e.g. <...>) were not seen in training.
- Labels derive partly from alignment-based annotation; an expert-curated test set is reported separately.

## Technical details
Backbone: <ESM-2 650M>, DAPT on <corpus>, multitask fine-tuning (uncertainty weighting), constrained decoding.
Hardware: <4 x H100, Jean Zay>, training time <h>, inference <ms/sequence on GPU / CPU>.
