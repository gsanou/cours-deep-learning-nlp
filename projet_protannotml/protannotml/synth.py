"""Exported from notebook pf01_donnees_imgt.ipynb — edit the notebook source (src/pf01_donnees_imgt.md), not this file."""

"""Synthetic IMGT-like reference directory and rearranged sequences (fictional data for development)."""
import numpy as np

from protannotml.imgt import REGIONS, cdr_positions, region_of, pos_key, ANCHORS

FR_RES = list("GSAVLTEQKPRDIFYN")
CDR_RES = list("YSGDNTAWRFHLK")
LOCI = {
    "IGH": dict(cdr1=[8, 8, 8, 9, 10], cdr2=[8, 8, 7, 10], fr_gaps=[10, 73], cdr3=(8, 22), cdr3_start="AR", j_anchor="W", c_len=98),
    "IGK": dict(cdr1=[6, 6, 7, 11, 12], cdr2=[3, 3], fr_gaps=[], cdr3=(7, 11), cdr3_start="QQ", j_anchor="F", c_len=102),
    "IGL": dict(cdr1=[8, 9, 6], cdr2=[3, 7], fr_gaps=[10], cdr3=(8, 13), cdr3_start="QS", j_anchor="F", c_len=100),
    "TRB": dict(cdr1=[5, 5, 6], cdr2=[6, 5], fr_gaps=[10, 73], cdr3=(10, 17), cdr3_start="AS", j_anchor="F", c_len=110),
}
SPECIES_DIVERGENCE = {"Homo sapiens": 0.04, "Macaca mulatta": 0.05, "Mus musculus": 0.16}
ANCHOR_AA = {23: "C", 41: "W", 104: "C"}


def _draw(rng, region):
    return str(rng.choice(CDR_RES if region.startswith("CDR") else FR_RES))


def mutate(seq, rng, rate_fr, rate_cdr, protect=(23, 41, 89, 104)):
    """Point mutations on a {position: aa} dict, CDR positions mutating faster; anchors protected."""
    out = dict(seq)
    for p, aa in seq.items():
        base = int(str(p).split(".")[0])
        if base in protect:
            continue
        region = region_of(p)
        if rng.random() < (rate_cdr if region.startswith("CDR") else rate_fr):
            new = _draw(rng, region)
            while new == aa:
                new = _draw(rng, region)
            out[p] = new
    return out


def _v_ancestor(rng):
    """Full-length V-REGION ancestor (positions 1-104, CDR1/CDR2 at maximum length)."""
    seq = {}
    for name, start, end in REGIONS[:5]:
        for p in range(start, end + 1):
            seq[str(p)] = ANCHOR_AA.get(p, _draw(rng, name))
    seq["89"] = str(rng.choice(list("LVIM")))
    return seq


def _restrict(seq, locus_cfg, cdr1_len, cdr2_len):
    keep = set(cdr_positions("CDR1", cdr1_len)) | set(cdr_positions("CDR2", cdr2_len))
    out = {}
    for p, aa in seq.items():
        region = region_of(p)
        if region in ("CDR1", "CDR2") and p not in keep:
            continue
        if int(p) in locus_cfg["fr_gaps"]:
            continue
        out[p] = aa
    return out


def to_gapped(seq, last=104):
    return "".join(seq.get(str(p), ".") for p in range(1, last + 1))


def make_reference(seed=0, n_families=4, genes_per_family=(3, 6), max_alleles=3, pseudo_rate=0.06):
    """Return a list of dicts: allele, gene, family, locus, species, functionality, positions {pos: aa}."""
    rng = np.random.default_rng(seed)
    refs, j_genes, c_domains = [], [], []
    for locus, cfg in LOCI.items():
        locus_anc = _v_ancestor(rng)
        for fam in range(1, n_families + 1):
            fam_anc = mutate(locus_anc, rng, 0.25, 0.55)
            cdr1, cdr2 = int(rng.choice(cfg["cdr1"])), int(rng.choice(cfg["cdr2"]))
            n_genes = int(rng.integers(*genes_per_family, endpoint=True))
            gene_ancs = [mutate(fam_anc, rng, 0.07, 0.25) for _ in range(n_genes)]
            for species, div in SPECIES_DIVERGENCE.items():
                for g, gene_anc in enumerate(gene_ancs, start=1):
                    if rng.random() < 0.15:          # gene absent from this species
                        continue
                    gene_seq = _restrict(mutate(gene_anc, rng, div, 2.5 * div), cfg, cdr1, cdr2)
                    gene = f"DEMO-{locus}V{fam}-{g}"
                    seen = []
                    for a in range(1, int(rng.integers(1, max_alleles, endpoint=True)) + 1):
                        seq = dict(gene_seq)
                        if a > 1:
                            for p in rng.choice([p for p in seq if int(p) not in ANCHOR_AA], size=int(rng.integers(1, 4)), replace=False):
                                seq[str(p)] = _draw(rng, region_of(p))
                        func = "F"
                        if rng.random() < pseudo_rate:
                            seq[str(rng.choice([p for p in seq if int(p) > 30]))] = "*"
                            func = "P"
                        refs.append(dict(allele=f"{gene}*{a:02d}", gene=gene, family=f"{locus}V{fam}", locus=locus,
                                         species=species, functionality=func, positions=seq))
                        seen.append(seq)
        for j in range(1, 5):                         # J genes: FR4-IMGT 118-128
            j_anc = [cfg["j_anchor"], "G"] + [str(rng.choice(FR_RES))] + ["G"] + [str(rng.choice(FR_RES)) for _ in range(7)]
            for species, div in SPECIES_DIVERGENCE.items():
                fr4 = {str(118 + i): aa for i, aa in enumerate(j_anc)}
                fr4 = mutate(fr4, rng, div, div, protect=(118, 119, 121))
                j_genes.append(dict(allele=f"DEMO-{locus}J{j}*01", gene=f"DEMO-{locus}J{j}", locus=locus, species=species, positions=fr4))
        c_anc = "".join(rng.choice(FR_RES, size=cfg["c_len"]))
        c_anc = c_anc[:24] + "C" + c_anc[25:80] + "C" + c_anc[81:]
        for species, div in SPECIES_DIVERGENCE.items():
            c = "".join(ch if (i in (24, 80) or rng.random() > div) else str(rng.choice(FR_RES)) for i, ch in enumerate(c_anc))
            c_domains.append(dict(name=f"DEMO-{locus}C", locus=locus, species=species, seq=c))
    return refs, j_genes, c_domains


def write_imgt_fasta(refs, path):
    """Write V-REGION alleles in the IMGT/GENE-DB AA-WithGaps layout (15 '|' fields)."""
    with open(path, "w") as f:
        for i, r in enumerate(refs, start=1):
            gapped = to_gapped(r["positions"])
            n_aa = sum(c != "." for c in gapped)
            fields = [f"DEMO{i:05d}", r["allele"], r["species"], r["functionality"], "V-REGION",
                      f"1..{3 * n_aa}", f"{3 * n_aa} nt", "1", " ", " ", " ", f"{n_aa} AA",
                      f"{n_aa}+{len(gapped) - n_aa}={len(gapped)}", " ", " "]
            f.write(">" + "|".join(fields) + "|\n" + gapped + "\n")


LINKER = "GGGGS" * 3
SIGNAL_RES = list("MKLLAVLLGLSAQPAMA")


def rearrange(rng, v, js, shm_rate):
    """V allele + numbered CDR3 + J (FR4) -> list of (position, aa), then somatic hypermutation."""
    cfg = LOCI[v["locus"]]
    cdr3_len = int(rng.integers(*cfg["cdr3"], endpoint=True))
    cdr3_pos = cdr_positions("CDR3", cdr3_len)
    cdr3_aa = list(cfg["cdr3_start"]) + [str(rng.choice(CDR_RES)) for _ in range(cdr3_len - 2)]
    j = js[int(rng.integers(len(js)))]
    seq = dict(v["positions"])
    seq.update(zip(cdr3_pos, cdr3_aa))
    seq.update(j["positions"])
    seq = mutate(seq, rng, shm_rate / 2, 1.5 * shm_rate, protect=(23, 41, 104, 118, 119, 121))
    return sorted(seq.items(), key=lambda kv: pos_key(kv[0])), j


def _v_part(rng, refs_by_ls, js_by_ls, locus, species, truncate):
    v = refs_by_ls[(locus, species)][int(rng.integers(len(refs_by_ls[(locus, species)])))]
    shm = float(rng.uniform(0, 0.12))
    residues, j = rearrange(rng, v, js_by_ls[(locus, species)], shm)
    if truncate:
        residues = residues[int(rng.integers(1, 6)):]
    meta = dict(type="V", locus=locus, species=species, v_allele=v["allele"], v_gene=v["gene"],
                j_gene=j["gene"], shm_rate=round(shm, 4))
    return residues, meta


def make_example(rng, refs_by_ls, js_by_ls, c_by_ls, kind, species):
    """One sequence with per-residue labels (domain BIO, region, IMGT position) and per-domain metadata."""
    seq, dom, reg, pos, domains = [], [], [], [], []

    def add_other(text, region="O"):
        for aa in text:
            seq.append(aa); dom.append("O"); reg.append(region); pos.append("-")

    def add_v(locus, truncate=False):
        residues, meta = _v_part(rng, refs_by_ls, js_by_ls, locus, species, truncate)
        meta["start"] = len(seq)
        for i, (p, aa) in enumerate(residues):
            seq.append(aa); dom.append("B-V" if i == 0 else "I-V"); reg.append(region_of(p)); pos.append(p)
        meta["end"] = len(seq)
        domains.append(meta)

    def add_c(locus):
        c = c_by_ls[(locus, species)]
        start = len(seq)
        for i, aa in enumerate(c["seq"]):
            seq.append(aa); dom.append("B-C" if i == 0 else "I-C"); reg.append("C"); pos.append("-")
        domains.append(dict(type="C", locus=locus, species=species, start=start, end=len(seq)))

    if rng.random() < 0.15:
        add_other("".join(rng.choice(SIGNAL_RES, size=int(rng.integers(15, 22)))))
    truncate = rng.random() < 0.15
    light = "IGK" if rng.random() < 0.6 else "IGL"
    if kind == "V":
        add_v(str(rng.choice(["IGH", "IGH", light, "TRB"])), truncate)
    elif kind == "scFv":
        add_v("IGH", truncate); add_other(LINKER); add_v(light)
    elif kind == "V+C":
        locus = str(rng.choice(["IGH", light, "TRB"]))
        add_v(locus, truncate); add_c(locus)
    return dict(kind=kind, species=species, seq="".join(seq), domain_tags=dom, regions=reg, imgt_pos=pos, domains=domains)


def make_dataset(refs, j_genes, c_domains, n=6000, seed=1, kinds=(("V", 0.5), ("scFv", 0.25), ("V+C", 0.25))):
    rng = np.random.default_rng(seed)
    refs_by_ls, js_by_ls = {}, {}
    for r in refs:
        if r["functionality"] == "F":
            refs_by_ls.setdefault((r["locus"], r["species"]), []).append(r)
    for j in j_genes:
        js_by_ls.setdefault((j["locus"], j["species"]), []).append(j)
    c_by_ls = {(c["locus"], c["species"]): c for c in c_domains}
    names, probs = zip(*kinds)
    species_list = list(SPECIES_DIVERGENCE)
    data = []
    for i in range(n):
        ex = make_example(rng, refs_by_ls, js_by_ls, c_by_ls, str(rng.choice(names, p=probs)), str(rng.choice(species_list)))
        ex["id"] = f"seq{i:05d}"
        data.append(ex)
    return data


def check_example(ex):
    from protannotml.imgt import REGION_NAMES
    n = len(ex["seq"])
    assert len(ex["domain_tags"]) == len(ex["regions"]) == len(ex["imgt_pos"]) == n
    for d in ex["domains"]:
        if d["type"] != "V":
            continue
        pos = ex["imgt_pos"][d["start"]:d["end"]]
        keys = [pos_key(p) for p in pos]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), "IMGT positions must increase strictly"
        regs = [REGION_NAMES.index(r) for r in ex["regions"][d["start"]:d["end"]]]
        assert regs == sorted(regs), "regions out of order"
        aa_at = dict(zip(pos, ex["seq"][d["start"]:d["end"]]))
        assert all(aa_at.get(str(a), c) == c for a, c in ((23, "C"), (41, "W"), (104, "C"))), "anchor mutated"
    return True


def equivalence_sets(refs, key="allele"):
    """Map each allele (keyed by species|allele) to the set of alleles or genes of the same locus and species
    with an identical V-REGION protein sequence."""
    from protannotml.imgt import allele_key
    sig = lambda r: (r["locus"], r["species"], tuple(sorted(r["positions"].items())))
    groups = {}
    for r in refs:
        groups.setdefault(sig(r), set()).add(r[key])
    return {allele_key(r["species"], r["allele"]): groups[sig(r)] for r in refs}


def save_jsonl(records, path):
    import json
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    import json
    with open(path) as f:
        return [json.loads(line) for line in f]
