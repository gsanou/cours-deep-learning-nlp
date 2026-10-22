# Module 17 — KG-RAG : graphes de connaissances RDF, SPARQL, Jena et Fuseki

> **Objectif** : quand les connaissances sont **structurées** (entités, relations, hiérarchies, nomenclatures), un **graphe de connaissances** répond mieux qu'une recherche dans du texte : réponses exactes, agrégations, chemins entre entités, raisonnement sur des hiérarchies. On apprend RDF, Turtle et SPARQL ; on code un **mini-moteur SPARQL** et un raisonneur RDFS pour comprendre ce qui se passe ; on construit un pipeline **text-to-SPARQL** avec validation et réparation ; on fait du **GraphRAG** (liaison d'entités, sous-graphe, verbalisation avec citations) ; puis on déploie le tout sur **Apache Jena Fuseki** et on l'expose à un agent via MCP.

| | |
|---|---|
| **Durée estimée** | 3 h 30 |
| **Prérequis** | Modules 15 (outils, MCP) et 16 (RAG) |
| **Librairies** | `numpy`, `scikit-learn` ; `rdflib`, `SPARQLWrapper`, `requests`, `mcp` (sections 7–9) ; Java 17+ et Apache Jena Fuseki |

> ⚠️ **Données fictives** : le graphe de démonstration s'inspire de l'organisation des connaissances en immunogénétique (gènes, allèles, loci, espèces, fonctionnalité), mais les **identifiants et valeurs sont inventés** (préfixe `DEMO`). Ce n'est pas un extrait d'IMGT ni d'aucune base réelle.

**Plan**

1. Pourquoi un graphe de connaissances avec un LLM ?
2. RDF en 15 minutes : triplets, IRI, littéraux, Turtle
3. Construire le graphe de démonstration
4. SPARQL, et un mini-moteur pour comprendre
5. Raisonner : RDFS et l'inférence
6. Text-to-SPARQL : générer, valider, réparer
7. GraphRAG : liaison d'entités, sous-graphe, verbalisation
8. Apache Jena et Fuseki en pratique
9. Brancher le graphe sur un agent (outil, MCP, LangChain)
10. Récapitulatif et exercices

```python
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.figsize": (7, 4), "axes.prop_cycle": plt.cycler(color=PALETTE),
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "lines.linewidth": 2, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
```

+++

## 1. Pourquoi un graphe de connaissances avec un LLM ?

| Question | RAG sur texte (module 16) | KG-RAG |
|---|---|---|
| « Que dit le protocole sur la CDR3 ? » | ✅ idéal | ❌ pas fait pour ça |
| « Combien d'allèles fonctionnels a le gène X chez l'espèce Y ? » | ⚠️ doit trouver et compter dans du texte, erreurs fréquentes | ✅ une requête, réponse exacte |
| « Quels gènes du locus IGH ont au moins un allèle pseudogène ? » | ❌ information dispersée | ✅ jointure |
| « Tous les gènes V » (y compris les sous-classes) | ❌ | ✅ inférence sur la hiérarchie |
| Traçabilité de la réponse | passages cités | triplets ou requête affichés : **vérifiables** |

Les deux se combinent : le graphe répond aux questions **factuelles et structurées**, le texte aux questions **explicatives**. Le LLM sert d'interface en langage naturel : il **traduit** la question en requête, ou **lit** un sous-graphe pour rédiger une réponse.

+++

## 2. RDF en 15 minutes

### 💡 Tout est triplet

RDF (*Resource Description Framework*, W3C) représente la connaissance comme un ensemble de **triplets** :
$$
(\text{sujet},\ \text{prédicat},\ \text{objet})
$$

- le **sujet** est une ressource identifiée par une **IRI** (un identifiant global, souvent une URL) ;
- le **prédicat** est une IRI qui nomme la relation ;
- l'**objet** est une IRI (une autre ressource) ou un **littéral** (chaîne, nombre, date, avec un type et éventuellement une langue).

Un ensemble de triplets est un **graphe orienté étiqueté**. Comme les identifiants sont globaux, deux graphes qui parlent de la même IRI se **fusionnent** simplement par union : c'est l'intérêt de RDF pour intégrer des sources hétérogènes.

### Turtle, la syntaxe lisible

```turtle
@prefix ex:   <http://example.org/demo#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

ex:DEMO-IGHV1-01  rdf:type        ex:VGene ;                 # ";" = same subject, new predicate
                  rdfs:label      "DEMO-IGHV1-01" ;
                  ex:inLocus      ex:IGH_Homo_sapiens ;
                  ex:hasAllele    ex:DEMO-IGHV1-01_01 , ex:DEMO-IGHV1-01_02 .   # "," = same predicate, new object

ex:DEMO-IGHV1-01_01  ex:functionality  ex:Functional ;
                     ex:cdr1Length     "8"^^xsd:integer .
```

### Le vocabulaire de base

| Vocabulaire | Rôle | Exemples |
|---|---|---|
| **RDF** | typage | `rdf:type` |
| **RDFS** | schéma léger | `rdfs:Class`, `rdfs:subClassOf`, `rdfs:subPropertyOf`, `rdfs:domain`, `rdfs:range`, `rdfs:label`, `rdfs:comment` |
| **OWL** | ontologies expressives | classes disjointes, cardinalités, propriétés inverses, transitives… |
| **SKOS** | thésaurus, terminologies | `skos:prefLabel`, `skos:altLabel`, `skos:broader` |
| **SHACL** | **validation** des données | « tout allèle doit avoir exactement une fonctionnalité » |

> 📎 Les ontologies de domaine existent pour beaucoup de disciplines (en biologie : Gene Ontology, et pour l'immunogénétique, IMGT-ONTOLOGY a été l'une des premières ontologies du domaine). **Réutiliser** un vocabulaire existant plutôt que d'inventer le sien rend le graphe interopérable.

+++

## 3. Construire le graphe de démonstration

On représente un terme par une chaîne : une IRI abrégée (`ex:VGene`), ou un **littéral** entre guillemets (`'"8"^^xsd:integer'`, `'"Homo sapiens"'`). Deux fonctions utilitaires convertissent les littéraux.

```python
def lit(value):
    if isinstance(value, bool):
        return f'"{str(value).lower()}"^^xsd:boolean'
    if isinstance(value, int):
        return f'"{value}"^^xsd:integer'
    return f'"{value}"'


def py(term):
    """Turn a literal term back into a Python value (IRIs are returned unchanged)."""
    m = re.fullmatch(r'"(.*)"(?:\^\^xsd:(\w+))?', term)
    if not m:
        return term
    v, dt = m.groups()
    return int(v) if dt == "integer" else (v == "true") if dt == "boolean" else v


rng = np.random.default_rng(0)
T = set()                                                     # the graph: a set of (s, p, o) triples
add = lambda s, p, o: T.add((s, p, o))

# --- schema (RDFS) ---
for c, parent in [("ex:VGene", "ex:Gene"), ("ex:JGene", "ex:Gene"), ("ex:Gene", "ex:GenomicFeature"), ("ex:Locus", "ex:GenomicFeature")]:
    add(c, "rdfs:subClassOf", parent)
for prop, dom, rng_, label in [("ex:hasAllele", "ex:Gene", "ex:Allele", "a pour allèle"),
                                ("ex:inLocus", "ex:Gene", "ex:Locus", "appartient au locus"),
                                ("ex:inSpecies", "ex:Locus", "ex:Species", "locus de l'espèce"),
                                ("ex:functionality", "ex:Allele", "ex:Functionality", "fonctionnalité"),
                                ("ex:cdr1Length", "ex:Allele", "xsd:integer", "longueur de la CDR1 (acides aminés)"),
                                ("ex:subgroup", "ex:VGene", "xsd:integer", "sous-groupe")]:
    add(prop, "rdf:type", "rdf:Property"); add(prop, "rdfs:domain", dom); add(prop, "rdfs:range", rng_); add(prop, "rdfs:label", lit(label))
for f in ["ex:Functional", "ex:ORF", "ex:Pseudogene"]:
    add(f, "rdf:type", "ex:Functionality"); add(f, "rdfs:label", lit(f.split(":")[1]))

# --- data (fictional identifiers) ---
species = {"ex:Homo_sapiens": "Homo sapiens", "ex:Mus_musculus": "Mus musculus"}
for sp, name in species.items():
    add(sp, "rdf:type", "ex:Species"); add(sp, "rdfs:label", lit(name))
    for locus in ["IGH", "IGK", "TRB"]:
        loc = f"ex:{locus}_{sp.split(':')[1]}"
        add(loc, "rdf:type", "ex:Locus"); add(loc, "ex:inSpecies", sp); add(loc, "rdfs:label", lit(f"{locus} ({name})"))
        for g in range(1, int(rng.integers(4, 7)) + 1):
            sub = int(rng.integers(1, 4))
            gname = f"DEMO-{locus}V{sub}-{g:02d}" + ("-mm" if sp == "ex:Mus_musculus" else "")
            gene = f"ex:{gname}"
            add(gene, "rdf:type", "ex:VGene"); add(gene, "rdfs:label", lit(gname)); add(gene, "ex:inLocus", loc); add(gene, "ex:subgroup", lit(sub))
            for a in range(1, int(rng.integers(1, 4)) + 1):
                allele = f"{gene}_{a:02d}"
                add(gene, "ex:hasAllele", allele); add(allele, "rdf:type", "ex:Allele"); add(allele, "rdfs:label", lit(f"{gname}*{a:02d}"))
                add(allele, "ex:functionality", rng.choice(["ex:Functional"] * 6 + ["ex:ORF"] * 2 + ["ex:Pseudogene"] * 2))
                add(allele, "ex:cdr1Length", lit(int(rng.choice([5, 6, 7, 8, 8, 9, 10]))))
        jgene = f"ex:DEMO-{locus}J1" + ("-mm" if sp == "ex:Mus_musculus" else "")
        add(jgene, "rdf:type", "ex:JGene"); add(jgene, "rdfs:label", lit(jgene.split(":")[1])); add(jgene, "ex:inLocus", loc)

print(f"{len(T)} triples")
for t in sorted(T)[:8]:
    print(t)
```

```python
PREFIXES = {"ex": "http://example.org/demo#", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#", "xsd": "http://www.w3.org/2001/XMLSchema#"}


def to_turtle(triples):
    lines = [f"@prefix {p}: <{iri}> ." for p, iri in PREFIXES.items()] + [""]
    by_subject = defaultdict(list)
    for s, p, o in sorted(triples):
        by_subject[s].append((p, o))
    for s, pos in by_subject.items():
        body = " ;\n    ".join(f"{p} {o}" for p, o in pos)
        lines.append(f"{s}\n    {body} .\n")
    return "\n".join(lines)

ttl = to_turtle(T)
with open("demo_kg.ttl", "w", encoding="utf-8") as fh:
    fh.write(ttl)
print(ttl[:900], "...")
print(f"\nwrote demo_kg.ttl ({len(ttl):,} characters): this file is loaded into Fuseki in section 8")
```

> ⚠️ Les IRI abrégées contenant `-` et `*` sont du Turtle valide seulement dans certaines limites (le `*` n'est pas permis dans un nom local) : c'est pourquoi les allèles ont des IRI en `_01` et le nom « humain » `…*01` en `rdfs:label`. Valide toujours un fichier Turtle avec un outil (`riot --validate demo_kg.ttl` de Jena, ou `rdflib`) avant de le charger.

+++

## 4. SPARQL, et un mini-moteur pour comprendre

### 💡 Une requête SPARQL est un motif de graphe

Une requête `SELECT` décrit un **motif** (*basic graph pattern*) : des triplets dont certains termes sont des **variables** (`?x`). Le moteur cherche toutes les façons d'**affecter** les variables pour que tous les triplets du motif existent dans le graphe. C'est une **jointure** sur les variables partagées.

```sparql
PREFIX ex: <http://example.org/demo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?geneLabel (COUNT(?allele) AS ?n)
WHERE {
  ?gene  ex:inLocus        ex:IGH_Homo_sapiens ;
         rdfs:label        ?geneLabel ;
         ex:hasAllele      ?allele .
  ?allele ex:functionality ex:Functional .
}
GROUP BY ?geneLabel
ORDER BY DESC(?n)
```

Les constructions à connaître : `FILTER` (conditions), `OPTIONAL` (jointure gauche), `UNION`, `GROUP BY` + agrégats (`COUNT`, `SUM`, `AVG`…), `ORDER BY`, `LIMIT`, les **chemins de propriétés** (`rdfs:subClassOf*` = zéro ou plusieurs pas), `ASK` (oui/non), `CONSTRUCT` (renvoyer un graphe), et `SERVICE` (interroger un autre point d'accès : requêtes **fédérées**).

### Un mini-moteur

Pour démystifier, voici un moteur qui exécute un **sous-ensemble** de SPARQL : `SELECT [DISTINCT]`, motifs de triplets séparés par `.` (avec `;` pour répéter le sujet), `FILTER` simples, chemin `rdfs:subClassOf*`, `LIMIT`. Ce n'est évidemment pas un vrai moteur (Jena, lui, optimise l'ordre des jointures, gère tout le langage et des milliards de triplets), mais **la logique est la même** : une recherche par retour sur trace (*backtracking*) qui étend les affectations de variables triplet par triplet.

```python
class MiniSPARQL:
    """Executes a small SPARQL subset over a set of (s, p, o) string triples."""

    def __init__(self, triples):
        self.T = set(triples)
        self.idx = defaultdict(set)                           # index by (position, value) for fast matching
        for t in self.T:
            for pos in range(3):
                self.idx[(pos, t[pos])].add(t)

    # ---------- parsing ----------
    def parse(self, query):
        q = re.sub(r"#[^\n]*", "", query)
        q = re.sub(r"PREFIX\s+\w+:\s*<[^>]+>", "", q, flags=re.I)
        if re.search(r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE)\b", q, re.I):
            raise ValueError("update operations are not allowed (read-only engine)")
        m = re.search(r"SELECT\s+(DISTINCT\s+)?(.*?)\s+WHERE\s*\{(.*)\}\s*(?:LIMIT\s+(\d+))?\s*$", q, re.S | re.I)
        if not m:
            raise ValueError("cannot parse: expected SELECT ... WHERE { ... } [LIMIT n]")
        distinct, sel, body, limit = m.groups()
        variables = re.findall(r"\?\w+", sel) if sel.strip() != "*" else None
        filters = re.findall(r"FILTER\s*\((.*?)\)\s*(?=\.|$|\?|\w+:|FILTER)", body + " ", re.S | re.I)
        body = re.sub(r"FILTER\s*\(.*?\)\s*(?=\.|$|\?|\w+:|FILTER)", " ", body + " ", flags=re.S | re.I)
        patterns, subject = [], None
        for stmt in [s.strip() for s in re.split(r"\s\.\s|\s\.$", body + " ") if s.strip()]:
            for k, part in enumerate(stmt.split(";")):
                toks = re.findall(r'"[^"]*"(?:\^\^\w+:\w+)?|\S+', part.strip())
                if not toks:
                    continue
                if k == 0:
                    subject, toks = toks[0], toks[1:]
                for o in " ".join(toks[1:]).split(","):
                    patterns.append((subject, toks[0], o.strip()))
        return {"vars": variables, "distinct": bool(distinct), "patterns": patterns, "filters": filters,
                "limit": int(limit) if limit else None}

    # ---------- evaluation ----------
    def subclasses_of(self, cls):
        out, stack = {cls}, [cls]
        while stack:
            c = stack.pop()
            for s, _, _ in self.idx[(2, c)]:
                if (s, "rdfs:subClassOf", c) in self.T and s not in out:
                    out.add(s); stack.append(s)
        return out

    def match(self, pattern, b):
        s, p, o = (b.get(x, x) for x in pattern)
        if p == "rdfs:subClassOf*":                          # property path: zero or more steps
            if not o.startswith("?"):
                for c in self.subclasses_of(o):
                    if s.startswith("?"):
                        yield {**b, s: c}
                    elif s == c:
                        yield b
            return
        cands = self.T
        for pos, val in enumerate((s, p, o)):
            if not val.startswith("?"):
                cands = cands & self.idx[(pos, val)] if cands is not self.T else self.idx[(pos, val)]
        for t in cands:
            nb, ok = dict(b), True
            for var, val in zip((s, p, o), t):
                if var.startswith("?"):
                    if nb.get(var, val) != val:
                        ok = False; break
                    nb[var] = val
            if ok:
                yield nb

    def check_filter(self, f, b):
        expr = f
        for var in sorted(set(re.findall(r"\?\w+", f)), key=len, reverse=True):
            if var not in b:
                return False
            expr = expr.replace(var, repr(py(b[var])))
        expr = re.sub(r'CONTAINS\s*\(\s*LCASE\s*\(\s*STR\s*\((.*?)\)\s*\)\s*,\s*("[^"]*")\s*\)', r"(\2.lower() in str(\1).lower())", expr, flags=re.I)
        expr = expr.replace("&&", " and ").replace("||", " or ").replace("!=", " != ")
        expr = re.sub(r"(?<![<>!=])=(?!=)", "==", expr)
        return bool(eval(expr, {"__builtins__": {}}, {"str": str}))   # tiny, trusted subset only (see note)

    def select(self, query):
        q = self.parse(query)
        results = [{}]
        for pat in sorted(q["patterns"], key=lambda t: sum(x.startswith("?") for x in t)):   # most selective first
            results = [nb for b in results for nb in self.match(pat, b)]
        results = [b for b in results if all(self.check_filter(f, b) for f in q["filters"])]
        rows = [tuple(b.get(v) for v in q["vars"]) for b in results] if q["vars"] else [tuple(b.values()) for b in results]
        if q["distinct"]:
            rows = list(dict.fromkeys(rows))
        rows.sort(key=lambda r: [str(x) for x in r])
        return rows[: q["limit"]] if q["limit"] else rows


kg = MiniSPARQL(T)
Q1 = """
PREFIX ex: <http://example.org/demo#>
SELECT ?geneLabel ?allele WHERE {
  ?gene ex:inLocus ex:IGH_Homo_sapiens ; rdfs:label ?geneLabel ; ex:hasAllele ?allele .
  ?allele ex:functionality ex:Functional .
}"""
rows = kg.select(Q1)
print(f"{len(rows)} functional alleles of IGH genes in Homo sapiens:")
counts = defaultdict(int)
for g, a in rows:
    counts[py(g)] += 1
for g, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {g:20s} {n}")
```

> ⚠️ Le `eval` restreint de `check_filter` est acceptable **ici** parce que les filtres sont validés en amont (section 6) et que c'est un jouet pédagogique. Un vrai moteur n'évalue jamais d'expression arbitraire : c'est une raison de plus pour utiliser Jena.

```python
Q2 = """SELECT ?alleleLabel ?len WHERE {
  ?gene ex:inLocus ?locus . ?locus ex:inSpecies ex:Mus_musculus .
  ?gene ex:hasAllele ?a . ?a rdfs:label ?alleleLabel ; ex:cdr1Length ?len .
  FILTER(?len >= 9)
} LIMIT 10"""
for r in kg.select(Q2):
    print(py(r[0]), py(r[1]))
```

+++

## 5. Raisonner : RDFS et l'inférence

Le graphe dit que `DEMO-IGHV1-01` est un `ex:VGene`, et que `ex:VGene` est une sous-classe de `ex:Gene`. Une requête « tous les `ex:Gene` » ne trouve **rien** si le moteur se contente des triplets écrits : l'information est **implicite**. Deux solutions :

1. **à la requête** : un chemin de propriété, `?g rdf:type/rdfs:subClassOf* ex:Gene` (ce que fait notre mini-moteur avec `rdfs:subClassOf*`) ;
2. **par matérialisation** : un **raisonneur** applique les règles de RDFS et **ajoute** les triplets déduits. Jena fournit des raisonneurs RDFS et OWL, configurables dans Fuseki.

Les règles RDFS principales (W3C) :

| Règle | Si… | alors… |
|---|---|---|
| rdfs9 | `?x rdf:type ?C` et `?C rdfs:subClassOf ?D` | `?x rdf:type ?D` |
| rdfs11 | `?C rdfs:subClassOf ?D` et `?D rdfs:subClassOf ?E` | `?C rdfs:subClassOf ?E` |
| rdfs2 | `?x ?p ?y` et `?p rdfs:domain ?C` | `?x rdf:type ?C` |
| rdfs3 | `?x ?p ?y` et `?p rdfs:range ?C` (et `?y` est une ressource) | `?y rdf:type ?C` |

On les applique jusqu'à **point fixe** (plus aucun triplet nouveau) : c'est le **chaînage avant**.

```python
def rdfs_closure(triples):
    T_ = set(triples)
    while True:
        new = set()
        sub = [(c, d) for c, p, d in T_ if p == "rdfs:subClassOf"]
        dom = {p: c for p, q, c in T_ if q == "rdfs:domain"}
        rng_ = {p: c for p, q, c in T_ if q == "rdfs:range" and not c.startswith("xsd:")}
        for c, d in sub:
            new |= {(c, "rdfs:subClassOf", e) for c2, e in sub if c2 == d}                       # rdfs11
            new |= {(x, "rdf:type", d) for x, p, c2 in T_ if p == "rdf:type" and c2 == c}         # rdfs9
        for x, p, y in T_:
            if p in dom:
                new.add((x, "rdf:type", dom[p]))                                                  # rdfs2
            if p in rng_ and not y.startswith('"'):
                new.add((y, "rdf:type", rng_[p]))                                                 # rdfs3
        if new <= T_:
            return T_
        T_ |= new

T_inf = rdfs_closure(T)
print(f"{len(T)} asserted triples -> {len(T_inf)} after RDFS inference (+{len(T_inf) - len(T)})")
Q_gene = "SELECT DISTINCT ?g WHERE { ?g rdf:type ex:Gene . }"
print("ex:Gene instances without inference:", len(MiniSPARQL(T).select(Q_gene)))
print("ex:Gene instances with inference   :", len(MiniSPARQL(T_inf).select(Q_gene)))
print("with a property path, no inference :", len(MiniSPARQL(T).select("SELECT DISTINCT ?g WHERE { ?g rdf:type ?c . ?c rdfs:subClassOf* ex:Gene . }")))
```

> ⚠️ RDFS **déduit**, il ne **valide** pas : si un triplet utilise `ex:hasAllele` sur une espèce par erreur, le domaine fera conclure que l'espèce est un `ex:Gene`. Pour **contrôler** la qualité des données, on utilise **SHACL** (formes et contraintes), pas RDFS.

+++

## 6. Text-to-SPARQL : générer, valider, réparer

### 💡 Le principe

On donne au LLM le **schéma** du graphe (classes, propriétés avec domaine et portée, exemples de valeurs) et la question ; il écrit une requête SPARQL ; l'application la **valide**, l'exécute, et renvoie les résultats (au LLM pour qu'il rédige la réponse, ou directement à l'utilisateur). Le schéma dans le prompt est crucial : sans lui, le modèle **invente** des prédicats plausibles.

```python
def schema_summary(triples):
    classes = sorted({o for s, p, o in triples if p == "rdf:type" and o.startswith("ex:")} | {s for s, p, o in triples if p == "rdfs:subClassOf"})
    props = sorted({s for s, p, o in triples if p == "rdf:type" and o == "rdf:Property"})
    lines = ["Classes:"] + [f"  {c}" + "".join(f" (subClassOf {o})" for s, p, o in triples if s == c and p == "rdfs:subClassOf") for c in classes]
    lines.append("Properties (domain -> range):")
    for p_ in props:
        d = next(o for s, p, o in triples if s == p_ and p == "rdfs:domain")
        r_ = next(o for s, p, o in triples if s == p_ and p == "rdfs:range")
        lab = py(next(o for s, p, o in triples if s == p_ and p == "rdfs:label"))
        lines.append(f"  {p_}: {d} -> {r_}   # {lab}")
    lines.append("Also: rdf:type, rdfs:label (human-readable names), rdfs:subClassOf.")
    lines.append("Example resources: ex:Homo_sapiens, ex:Mus_musculus, ex:IGH_Homo_sapiens, ex:Functional, ex:ORF, ex:Pseudogene")
    return "\n".join(lines)

SCHEMA = schema_summary(T)
print(SCHEMA)
```

### La validation : ne jamais exécuter une requête générée sans contrôle

| Contrôle | Pourquoi |
|---|---|
| syntaxe | le modèle peut produire une requête mal formée |
| **lecture seule** | refuser `INSERT`, `DELETE`, `DROP`… (et utiliser un point d'accès en lecture seule) |
| préfixes et prédicats **connus** | détecter les hallucinations de schéma (`ex:hasSpecies` n'existe pas) |
| `LIMIT` imposé | éviter les requêtes qui renvoient des millions de lignes |
| délai maximal | éviter les requêtes coûteuses (Fuseki permet de configurer un *timeout*) |
| résultat vide | souvent signe d'une erreur ; à signaler au modèle pour réparation |

En cas d'échec, on renvoie au modèle un **message d'erreur précis** (« le prédicat ex:hasSpecies n'existe pas ; prédicats disponibles : … ») et il corrige : c'est la boucle **générer → valider → réparer**.

```python
KNOWN_PREDICATES = {p for _, p, _ in T} | {"rdfs:subClassOf*"}
KNOWN_PREFIXES = set(PREFIXES)


def validate_sparql(query, engine, max_limit=100):
    errors = []
    if re.search(r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE)\b", query, re.I):
        return None, ["update operations are forbidden: this endpoint is read-only"]
    try:
        parsed = engine.parse(query)
    except ValueError as e:
        return None, [f"syntax error: {e}"]
    for s, p, o in parsed["patterns"]:
        for term in (s, p, o):
            m = re.match(r"(\w+):", term)
            if m and not term.startswith('"') and m.group(1) not in KNOWN_PREFIXES:
                errors.append(f"unknown prefix '{m.group(1)}:' in {term}")
        if not p.startswith("?") and p not in KNOWN_PREDICATES:
            errors.append(f"unknown predicate {p}; available: {sorted(x for x in KNOWN_PREDICATES if x.startswith('ex:'))}")
    if errors:
        return None, errors
    if not re.search(r"LIMIT\s+\d+", query, re.I):
        query = query.rstrip() + f" LIMIT {max_limit}"
    return query, []


class MockText2SPARQL:
    """Simulated LLM: returns canned queries, one of them with a hallucinated predicate, and repairs after feedback."""
    CANNED = {
        "allèles fonctionnels du locus IGH humain": ("SELECT ?a WHERE { ?g ex:inLocus ex:IGH_Homo_sapiens ; ex:hasAllele ?a . ?a ex:functionality ex:Functional . }", None),
        "gènes de la souris": ("SELECT DISTINCT ?label WHERE { ?g ex:hasSpecies ex:Mus_musculus ; rdfs:label ?label . }",
                               "SELECT DISTINCT ?label WHERE { ?g ex:inLocus ?l ; rdfs:label ?label . ?l ex:inSpecies ex:Mus_musculus . }"),
        "allèles pseudogènes avec une CDR1 de plus de 8": ("SELECT ?lab ?len WHERE { ?a ex:functionality ex:Pseudogene ; ex:cdr1Length ?len ; rdfs:label ?lab . FILTER(?len > 8) }", None),
        "supprime les pseudogènes": ("DELETE WHERE { ?a ex:functionality ex:Pseudogene . }", None),
    }

    def generate(self, question, schema, feedback=None):
        key = next(k for k in self.CANNED if k in question)
        first, repaired = self.CANNED[key]
        return repaired if (feedback and repaired) else first


def text_to_sparql(question, llm, engine, max_repairs=2):
    feedback = None
    for attempt in range(max_repairs + 1):
        query = llm.generate(question, SCHEMA, feedback)
        safe_query, errors = validate_sparql(query, engine)
        print(f"  attempt {attempt}: {query.strip()[:110]}")
        if errors:
            feedback = "; ".join(errors)
            print(f"    REJECTED: {feedback[:150]}")
            if "read-only" in feedback:
                return None, []
            continue
        rows = engine.select(safe_query)
        print(f"    OK: {len(rows)} rows")
        return safe_query, rows
    return None, []

llm = MockText2SPARQL()
for qn in ["Liste les allèles fonctionnels du locus IGH humain.", "Quels sont les gènes de la souris ?",
           "Quels allèles pseudogènes avec une CDR1 de plus de 8 acides aminés ?", "Merci de supprime les pseudogènes du graphe."]:
    print(f"\nQ: {qn}")
    q_ok, rows = text_to_sparql(qn, llm, MiniSPARQL(T_inf))
    for r in rows[:4]:
        print("     ", tuple(py(x) for x in r))
```

> 💡 **Améliorer la génération** : inclure dans le prompt quelques **exemples question → requête** (few-shot) choisis par similarité avec la question (un mini-RAG sur une bibliothèque de requêtes validées), donner les **IRI exactes** des entités mentionnées (liaison d'entités, section 7), et demander au modèle d'**expliquer** sa requête en une phrase, ce qui aide l'utilisateur à vérifier qu'elle correspond à sa question.

+++

## 7. GraphRAG : liaison d'entités, sous-graphe, verbalisation

Le text-to-SPARQL demande au modèle d'écrire une requête exacte. L'autre approche, plus tolérante : **récupérer un sous-graphe** autour des entités de la question et le donner au LLM **sous forme de texte**, comme les passages d'un RAG.

1. **Liaison d'entités** : retrouver les IRI des entités mentionnées, malgré les variantes d'écriture (« IGHV1 01 », « ighv1-01 ») ;
2. **Extraction de sous-graphe** : les triplets à 1 ou 2 sauts de ces entités, **sans traverser les classes** (`ex:VGene` relie tous les gènes : passer par elle ramènerait tout le graphe) ;
3. **Verbalisation** : transformer les triplets en phrases numérotées, citables.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_CLASSES = {"ex:VGene", "ex:JGene", "ex:Allele", "ex:Locus", "ex:Species"}
instances = {s for s, p, o in T if p == "rdf:type" and o in DATA_CLASSES}
labels = {s: py(o) for s, p, o in T if p == "rdfs:label"}
linkable = {s: lab for s, lab in labels.items() if s in instances}          # link mentions to DATA entities, not schema terms
ids, names = list(linkable), list(linkable.values())
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True).fit(names)
L = char_vec.transform(names)


def link_entities(text, threshold=0.55, k=3):
    """Fuzzy entity linking with character n-grams (robust to spaces, case, separators)."""
    cands = re.findall(r"[A-Za-z][\w\-\*]+(?:\s\d+)?", text)
    out = []
    for c in cands:
        sims = (L @ char_vec.transform([c]).T).toarray().ravel()
        best = np.argsort(-sims)[:k]
        if sims[best[0]] >= threshold:
            out.append((c, ids[best[0]], round(float(sims[best[0]]), 2)))
    return out

some_gene = next(s for s, p, o in T if p == "rdf:type" and o == "ex:VGene" and "Homo" in "".join(oo for ss, pp, oo in T if ss == s and pp == "ex:inLocus"))
mention = labels[some_gene].lower().replace("-", " ", 1)
print("mention:", mention, "->", link_entities(f"Quelle est la fonctionnalité des allèles de {mention} ?"))
```

```python
def subgraph(seeds, triples, hops=2, max_triples=40):
    frontier, seen, out = set(seeds), set(seeds), []
    for _ in range(hops):
        nxt = set()
        for s, p, o in sorted(triples):
            if (s in frontier or o in frontier) and p not in ("rdfs:subClassOf", "rdfs:label") and (s, p, o) not in out:
                out.append((s, p, o))
                if p == "rdf:type":
                    continue                                   # never expand through classes: they are hubs linking everything
                for x in (s, o):
                    if not x.startswith('"') and x not in seen:
                        nxt.add(x); seen.add(x)
        frontier = nxt
    return out[:max_triples]


PRED_FR = {"rdf:type": "est un(e)", "rdfs:label": "a pour nom", "ex:hasAllele": "a pour allèle", "ex:inLocus": "appartient au locus",
           "ex:inSpecies": "est un locus de l'espèce", "ex:functionality": "a pour fonctionnalité", "ex:cdr1Length": "a une CDR1 de longueur",
           "ex:subgroup": "appartient au sous-groupe"}


def verbalize(triples):
    name = lambda x: str(py(x)) if x.startswith('"') else str(labels.get(x, x.split(":")[-1]))
    return [f"[T{i}] {name(s)} {PRED_FR.get(p, p)} {name(o)}." for i, (s, p, o) in enumerate([t for t in triples if t[1] != "rdfs:label"], 1)]

linked = [iri for _, iri, _ in link_entities(f"allèles de {mention}")]
facts = verbalize(subgraph(linked, T, hops=2, max_triples=30))
print("\n".join(facts[:14]))
prompt = ("Réponds à partir des faits ci-dessous uniquement, en citant leurs identifiants [Tn]. "
          "Si les faits ne suffisent pas, dis-le.\n\n" + "\n".join(facts) +
          f"\n\nQuestion : quelle est la fonctionnalité des allèles de {labels[linked[0]]} ?")
print(f"\n(prompt of {len(prompt.split())} words ready for the LLM)")
```

| | Text-to-SPARQL | Sous-graphe + verbalisation |
|---|---|---|
| exactitude, agrégations (compter, filtrer) | ✅ le moteur calcule | ⚠️ le LLM compte dans le texte (erreurs) |
| tolérance aux questions vagues | ⚠️ | ✅ |
| explicabilité | la requête | les triplets cités |
| coût | 1–3 appels + requête | contexte plus long |
| risque | requête fausse mais valide (réponse fausse « exacte ») | sous-graphe incomplet |

> 📎 **« GraphRAG » de Microsoft** (Edge et al., 2024) désigne une approche différente et complémentaire : un LLM **construit** un graphe d'entités à partir de documents textuels, le partitionne en communautés et rédige un résumé par communauté, pour répondre à des questions **globales** sur un corpus (« quels sont les grands thèmes ? »). La **construction de graphes par LLM** (extraction de triplets depuis du texte) est puissante mais bruitée : contraindre l'extraction à une ontologie et **valider par SHACL** avant intégration.

+++

## 8. Apache Jena et Fuseki en pratique

**Apache Jena** est un framework Java pour le web sémantique : stockage (TDB2), moteur SPARQL (ARQ), raisonneurs RDFS/OWL, validation SHACL, outils en ligne de commande. **Fuseki** est son **serveur SPARQL** : il expose des jeux de données via HTTP (requêtes, mises à jour, chargement de fichiers) et une interface web.

### Installer et lancer Fuseki

```bash
# Requires Java 17+. Download "apache-jena-fuseki-X.Y.Z" from https://jena.apache.org/download/ and unzip it.
cd apache-jena-fuseki-X.Y.Z

# In-memory dataset named /ds (data lost at shutdown), updates allowed:
./fuseki-server --update --mem /ds

# Persistent TDB2 database, read-only endpoint (recommended for an LLM-facing service):
./fuseki-server --tdb2 --loc=databases/demo /ds
# UI: http://localhost:3030   |  endpoints for /ds: /ds/sparql (query), /ds/update, /ds/data (Graph Store Protocol), /ds/get

# Docker: Apache provides a Dockerfile (jena-fuseki-docker) to build an image for a given Jena version:
#   docker build --build-arg JENA_VERSION=<version> -t fuseki .
#   docker run -i --rm -p 3030:3030 -t fuseki --mem /ds
```

### Charger, valider, interroger

```bash
# Validate the Turtle file first (Jena command-line tools, from the "apache-jena" distribution)
riot --validate demo_kg.ttl

# Bulk-load into a TDB2 database (fast, offline), then serve it read-only
tdb2.tdbloader --loc databases/demo demo_kg.ttl

# Or upload into a running server (Graph Store Protocol), into the default graph:
curl -X POST -H "Content-Type: text/turtle" --data-binary @demo_kg.ttl "http://localhost:3030/ds/data?default"

# Query over HTTP
curl -s http://localhost:3030/ds/sparql -H "Accept: application/sparql-results+json" \
     --data-urlencode 'query=PREFIX ex: <http://example.org/demo#> SELECT (COUNT(*) AS ?n) WHERE { ?a a ex:Allele }'
```

```python noexec
# pip install SPARQLWrapper rdflib requests
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint = SPARQLWrapper("http://localhost:3030/ds/sparql")
endpoint.setReturnFormat(JSON)
endpoint.setTimeout(10)
endpoint.setQuery("""
PREFIX ex: <http://example.org/demo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?geneLabel (COUNT(?a) AS ?n) WHERE {
  ?g ex:inLocus ex:IGH_Homo_sapiens ; rdfs:label ?geneLabel ; ex:hasAllele ?a .
  ?a ex:functionality ex:Functional .
} GROUP BY ?geneLabel ORDER BY DESC(?n)
""")
for b in endpoint.query().convert()["results"]["bindings"]:
    print(b["geneLabel"]["value"], b["n"]["value"])
```

```python noexec
# rdflib: parse, validate and query locally (handy for tests, and to syntax-check LLM-generated SPARQL before sending it to Fuseki)
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

g = Graph().parse("demo_kg.ttl", format="turtle")
print(len(g), "triples")
try:
    pq = prepareQuery("SELECT ?a WHERE { ?a ex:functionality ex:Functional }", initNs={"ex": "http://example.org/demo#"})
    print(len(list(g.query(pq))), "functional alleles")
except Exception as e:                       # rdflib raises a ParseException on malformed SPARQL
    print("invalid SPARQL:", e)
```

### Inférence côté serveur

Fuseki peut servir un jeu de données **avec raisonneur** via un fichier de configuration (« assembleur ») qui empile un modèle d'inférence (par exemple le raisonneur RDFS de Jena) au-dessus du stockage TDB2. Les requêtes voient alors les triplets déduits (section 5) sans les stocker. Pour de gros graphes, on préfère souvent **matérialiser** l'inférence une fois (hors ligne) et servir le résultat, ou utiliser des chemins de propriétés dans les requêtes.

### Des données réelles : Wikidata, UniProt

```python noexec
wd = SPARQLWrapper("https://query.wikidata.org/sparql", agent="cours-kg-rag/0.1 (contact: you@example.org)")
wd.setReturnFormat(JSON)
wd.setQuery("""
SELECT ?protein ?proteinLabel WHERE {
  ?protein wdt:P31 wd:Q8054 ;          # instance of: protein
           wdt:P703 wd:Q15978631 .      # found in taxon: Homo sapiens
  SERVICE wikibase:label { bd:serviceParam wikibase:language "fr,en". }
} LIMIT 5""")
print([b["proteinLabel"]["value"] for b in wd.query().convert()["results"]["bindings"]])
# UniProt also exposes a public SPARQL endpoint (https://sparql.uniprot.org/sparql) with its own ontology.
```

+++

## 9. Brancher le graphe sur un agent

La façon la plus propre : un **serveur MCP** (module 15) qui expose le **schéma** comme ressource et une requête SPARQL **en lecture seule** comme outil. N'importe quel agent (Claude Desktop, Claude Code, LangGraph…) peut alors interroger le graphe, et le serveur centralise les contrôles (validation, `LIMIT`, délai).

```python noexec
# file: kg_server.py      pip install "mcp>=2" SPARQLWrapper rdflib
from mcp.server import MCPServer
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.plugins.sparql import prepareQuery
import re

FUSEKI = "http://localhost:3030/ds/sparql"                      # a READ-ONLY dataset (no /update exposed)
NS = {"ex": "http://example.org/demo#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xsd": "http://www.w3.org/2001/XMLSchema#"}
PREFIX_BLOCK = "\n".join(f"PREFIX {k}: <{v}>" for k, v in NS.items())
mcp = MCPServer("demo-kg")


@mcp.resource("kg://schema")
def schema() -> str:
    """Classes and properties of the knowledge graph (read this before writing SPARQL)."""
    return open("schema.txt", encoding="utf-8").read()             # e.g. the output of schema_summary()


@mcp.tool()
def sparql_select(query: str, limit: int = 50) -> dict:
    """Run a read-only SPARQL SELECT query on the knowledge graph. Prefixes ex:, rdf:, rdfs:, xsd: are predeclared.
    Always read the kg://schema resource first; only use predicates listed there."""
    if re.search(r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE)\b", query, re.I):
        return {"error": "read-only endpoint: updates are not allowed"}
    full = PREFIX_BLOCK + "\n" + query
    try:
        prepareQuery(full)                                         # syntax check before hitting the server
    except Exception as e:
        return {"error": f"SPARQL syntax error: {e}"}
    if not re.search(r"LIMIT\s+\d+", query, re.I):
        full += f"\nLIMIT {min(limit, 200)}"
    ep = SPARQLWrapper(FUSEKI); ep.setReturnFormat(JSON); ep.setTimeout(15); ep.setQuery(full)
    res = ep.query().convert()
    rows = [{k: v["value"] for k, v in b.items()} for b in res["results"]["bindings"]]
    return {"rows": rows, "n": len(rows)}


if __name__ == "__main__":
    mcp.run()
```

```python noexec
# Use it from LangChain (or declare it in Claude Desktop / Claude Code: `claude mcp add demo-kg -- python kg_server.py`)
import asyncio
from pathlib import Path
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def ask(question):
    async with MCPAdapter(Path("kg_server.py")) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent("anthropic:claude-sonnet-5-5", tools,
                             system_prompt="Tu réponds aux questions sur le graphe de connaissances. Lis d'abord le schéma, "
                                           "écris des requêtes SPARQL avec les seuls prédicats du schéma, montre la requête utilisée.")
        out = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        return out["messages"][-1].content

print(asyncio.run(ask("Combien d'allèles fonctionnels compte chaque gène du locus IGH humain ?")))
```

**Combiner texte et graphe** : un agent qui dispose à la fois d'un outil `search_docs` (module 16) et d'un outil `sparql_select` peut répondre à « Que dit le protocole sur les ORF, et combien d'allèles ORF avons-nous dans IGH ? » en appelant les deux. C'est le schéma du module 18 (agents spécialisés coordonnés).

+++

## 10. Récapitulatif

| Notion | À retenir |
|---|---|
| RDF | triplets (sujet, prédicat, objet) ; IRI globales ; littéraux typés ; Turtle |
| Vocabulaires | RDFS (classes, domaines, portées), OWL, SKOS ; réutiliser l'existant ; SHACL pour valider |
| SPARQL | motifs de graphe = jointures ; FILTER, OPTIONAL, GROUP BY, chemins de propriétés |
| Inférence | règles RDFS jusqu'au point fixe, ou chemins `subClassOf*` à la requête |
| Text-to-SPARQL | schéma dans le prompt ; **valider** (lecture seule, prédicats connus, LIMIT, délai) ; réparer avec l'erreur |
| GraphRAG | liaison d'entités floue, sous-graphe à k sauts, verbalisation citable |
| Fuseki | `fuseki-server --tdb2 --loc=… /ds` ; `/ds/sparql`, `/ds/data` ; `tdb2.tdbloader`, `riot --validate` |
| Agents | serveur MCP : ressource schéma + outil SPARQL en lecture seule |

### 🏋️ Exercices

1. **OPTIONAL.** Étends `MiniSPARQL` pour gérer `OPTIONAL { … }` (jointure gauche) et teste-le : « tous les gènes et, s'ils en ont, leurs allèles ORF ».
2. **SHACL.** Écris une forme SHACL qui impose que tout `ex:Allele` ait exactement une `ex:functionality` et une `ex:cdr1Length` entière entre 1 et 20. Valide `demo_kg.ttl` avec `pyshacl` ou `shacl validate` de Jena, après y avoir introduit une erreur.
3. **Few-shot dynamique.** Construis une bibliothèque de 15 paires (question, requête validée), et pour chaque nouvelle question, place dans le prompt les 3 paires les plus proches (BM25 du module 16).
4. **Évaluer le text-to-SPARQL.** Pour 20 questions, écris la requête de référence et compare les **résultats** (pas le texte des requêtes) de la requête générée : exactitude d'ensemble (*execution accuracy*).
5. **Fuseki.** Charge `demo_kg.ttl` dans Fuseki (TDB2, lecture seule), lance `kg_server.py`, branche-le dans Claude Code, et pose trois questions qui exigent une jointure sur trois triplets.

<details><summary>Indications</summary>

1. Pour chaque affectation courante, chercher les extensions du motif optionnel ; s'il n'y en a aucune, garder l'affectation telle quelle (variables optionnelles non liées).
4. Comparer des ensembles de lignes (ordre ignoré sauf `ORDER BY`) : deux requêtes différentes peuvent être également justes.
</details>

**➡️ Module suivant : multi-agents, LangChain et LangGraph** (graphes d'état, superviseur, transferts entre agents, points de contrôle, humain dans la boucle).
