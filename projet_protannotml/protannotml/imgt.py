"""Exported from notebook pf01_donnees_imgt.ipynb — edit the notebook source (src/pf01_donnees_imgt.md), not this file."""

"""IMGT unique numbering for V domains: regions, CDR gap/insertion rules, FASTA headers, AIRR rows."""
from collections import defaultdict
from dataclasses import dataclass

REGIONS = [("FR1", 1, 26), ("CDR1", 27, 38), ("FR2", 39, 55), ("CDR2", 56, 65),
           ("FR3", 66, 104), ("CDR3", 105, 117), ("FR4", 118, 128)]
REGION_NAMES = [r for r, _, _ in REGIONS]
ANCHORS = {23: "1st-CYS", 41: "CONSERVED-TRP", 89: "hydrophobic", 104: "2nd-CYS", 118: "J-TRP/PHE"}
CDR_GAP_ORDER = {
    "CDR1": [33, 32, 34, 31, 35, 30, 36],
    "CDR2": [61, 60, 62, 59, 63, 58, 64, 57, 65, 56],
    "CDR3": [111, 112, 110, 113, 109, 114, 108, 115, 107, 116, 106, 117, 105],
}


def parse_pos(p):
    """'111.2' -> (111, 2); '45' or 45 -> (45, 0)."""
    s = str(p)
    if "." in s:
        base, ins = s.split(".")
        return int(base), int(ins)
    return int(s), 0


def pos_key(p):
    """Sort key giving the order of IMGT positions along the chain (111, 111.1, 111.2, ..., 112.2, 112.1, 112)."""
    base, ins = parse_pos(p)
    if ins == 0:
        return (base, 0, 0)
    if base == 111:
        return (111, 1, ins)
    if base == 112:
        return (111, 2, -ins)
    raise ValueError(f"insertions are only defined at 111/112 here, got {p}")


def region_of(pos):
    base, _ = parse_pos(pos)
    for name, start, end in REGIONS:
        if start <= base <= end:
            return name
    raise ValueError(f"position {pos} outside the V domain (1-128)")


def cdr_positions(region, length):
    """Occupied IMGT positions (chain order) of a CDR of a given length."""
    start, end = next((s, e) for r, s, e in REGIONS if r == region)
    full = list(range(start, end + 1))
    if length <= len(full):
        n_gaps = len(full) - length
        order = CDR_GAP_ORDER[region]
        if n_gaps > len(order):
            raise ValueError(f"{region} of length {length} is shorter than this table allows")
        gaps = set(order[:n_gaps])
        return [str(p) for p in full if p not in gaps]
    if region != "CDR3":
        raise ValueError(f"{region} longer than {len(full)} is not handled here")
    insertions, k = [], 1
    while len(insertions) < length - len(full):
        insertions.append(f"112.{k}")
        if len(insertions) < length - len(full):
            insertions.append(f"111.{k}")
        k += 1
    return sorted([str(p) for p in full] + insertions, key=pos_key)


IMGT_FIELDS = ["accession", "allele", "species", "functionality", "label", "positions", "n_nt",
               "codon_start", "added_5", "added_3", "corrections", "n_aa", "n_chars", "partial", "revcomp"]


@dataclass
class ImgtRecord:
    fields: dict
    seq: str

    @property
    def allele(self):
        return self.fields["allele"]

    @property
    def gene(self):
        return self.fields["allele"].split("*")[0]

    @property
    def species(self):
        return self.fields["species"]

    @property
    def locus(self):
        return self.gene[:3] if self.gene[:2] in ("IG", "TR") else self.gene.split("-")[1][:3]


def parse_imgt_header(header):
    parts = [p.strip() for p in header.lstrip(">").split("|")]
    parts += [""] * (len(IMGT_FIELDS) - len(parts))
    return dict(zip(IMGT_FIELDS, parts))


def read_imgt_fasta(lines):
    """Yield ImgtRecord from an iterable of lines (an open file or a list of strings)."""
    header, chunks = None, []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield ImgtRecord(parse_imgt_header(header), "".join(chunks))
            header, chunks = line, []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        yield ImgtRecord(parse_imgt_header(header), "".join(chunks))


def gapped_to_positions(gapped, first_position=1):
    """IMGT-gapped V-REGION (one char per IMGT position, '.' = gap) -> [(position, aa)]."""
    return [(str(i + first_position), aa) for i, aa in enumerate(gapped) if aa != "."]


def allele_key(species, allele):
    """Allele names are only unique within a species: always key alleles by (species, name)."""
    return f"{species}|{allele}"


AIRR_REGION_COLUMNS = [("FR1", "fwr1_aa"), ("CDR1", "cdr1_aa"), ("FR2", "fwr2_aa"), ("CDR2", "cdr2_aa"),
                       ("FR3", "fwr3_aa"), ("CDR3", "cdr3_aa"), ("FR4", "fwr4_aa")]


def airr_row_to_v_domain(row):
    """AIRR rearrangement row (dict) -> list of (IMGT position, aa) for the V domain, or raise ValueError.

    FR1-FR3 / CDR1-CDR2 positions come from the IMGT-gapped `sequence_alignment_aa` (assumed to start at
    position 1, missing 5' positions shown as '.'); CDR3 is numbered with the IMGT length rule; FR4 from 118.
    """
    gapped = row["sequence_alignment_aa"]
    v_part = gapped[:104]
    residues = [(str(i + 1), aa) for i, aa in enumerate(v_part) if aa not in ".-"]
    cdr3, fr4 = row.get("cdr3_aa") or "", row.get("fwr4_aa") or ""
    residues += list(zip(cdr_positions("CDR3", len(cdr3)), cdr3))
    residues += [(str(118 + i), aa) for i, aa in enumerate(fr4)]
    rebuilt = defaultdict(str)
    for p, aa in residues:
        rebuilt[region_of(p)] += aa
    for region, col in AIRR_REGION_COLUMNS:
        if (row.get(col) or "") != rebuilt.get(region, ""):
            raise ValueError(f"{row.get('sequence_id')}: {region} mismatch ({row.get(col)!r} vs {rebuilt.get(region, '')!r})")
    return residues
