# Parcours raccourci centré sur ProtAnnotML

> Les 22 modules et 6 notebooks du fil rouge représentent environ 90 heures. Pour ProtAnnotML, tu n'as pas besoin de tout, ni dans l'ordre du sommaire. Voici un parcours d'environ **40 heures** en quatre étapes, avec pour chaque module ce qu'il faut **faire**, ce qu'on peut **survoler** et ce qu'on peut **sauter**, puis deux parcours optionnels (structure, et assistant IMGT / KG-RAG).

| | |
|---|---|
| **Parcours principal** | ≈ 40 h, en 4 étapes (par exemple 4 semaines à 10 h) |
| **Hypothèse** | tu connais déjà les bases du ML et de Python, et très bien la numérotation IMGT |
| **Principe** | chaque étape se termine par un livrable concret pour le projet |

+++

## Vue d'ensemble

| Étape | Contenu | Durée | Livrable |
|---|---|---|---|
| 1. Les outils | m00 (survol), m01, m06, m07 (partiel) | ≈ 9 h | tu sais écrire et déboguer un Transformer et un affinage HF |
| 2. Le cœur du projet | m08, m09, m10 (partiel), m13 (partiel) | ≈ 10 h | protocole expérimental rédigé (découpages, métriques, ablations) |
| 3. Le fil rouge | pf01 → pf06 | ≈ 13 h | données réelles converties, références calculées, premier modèle entraîné |
| 4. Comprendre et publier | m19, pf05 revisité, lectures | ≈ 8 h | figures d'interprétation et tableau de résultats avec IC |
| Optionnel A | m20 (structure), m11, m12 | ≈ 8 h | ablation SaProt / structure, petit modèle distillé pour le service web |
| Optionnel B | m15–m18, m21 | ≈ 20 h | assistant IMGT : KG-RAG sur IMGT/mAb-KG, outils MCP, agents |

+++

## Étape 1 — Les outils (≈ 9 h)

| Module | Faire | Survoler | Sauter |
|---|---|---|---|
| **m00 Fondements** (3–4 h → 1 h) | §7 rétropropagation, §10 régularisation | §3–6 si tu connais les activations, pertes, optimiseurs | §1–2 |
| **m01 PyTorch** (2–3 h → 2 h 30) | §3 autograd, §5 padding et masques, §6 boucle robuste, §7 débogage | §2 tenseurs | — |
| **m06 Transformer** (4 h → 3 h) | §2–5 attention, masques, multi-têtes, positions (RoPE), §7 résiduels/LayerNorm, §9 coûts | §8 Transformer NumPy | §10–11 mini-GPT génératif (inutile pour un encodeur) |
| **m07 HF** (3 h → 2 h 30) | §2 objectifs (MLM), §4 écosystème, §7 étiquetage de tokens (alignement tokens/étiquettes) | §3 lois d'échelle | §5 décodage, §9 assistants |

**À sauter entièrement pour ProtAnnotML** : m02 (tokenisation du texte : les protéines ont un token par résidu), m03 (word2vec), m04 (RNN), m05 (seq2seq). Ils restent utiles pour la culture générale ; m05 §3–4 (l'attention comme dictionnaire flou) est une bonne lecture de 20 minutes si l'attention te semble encore abstraite.

+++

## Étape 2 — Le cœur du projet (≈ 10 h)

| Module | Faire | Survoler | Sauter |
|---|---|---|---|
| **m08 Adaptation et évaluation** (3 h 30, tout) | §2 DAPT/TAPT, §3 LoRA, §4 multi-tâche et incertitude, §5 fuite par similarité, §6 IC et comparaison, §7 calibration, §8 check-list | — | — |
| **m09 PLM** (3 h → 2 h 30) | §3 co-évolution, §4–5 simulateur de domaines V et décodage structuré, §6–7 ESM-2 HF et affinage, §9 pièges | §2 tokenisation | §8 effets de mutations (hors sujet pour l'annotation, sauf curiosité) |
| **m10 NER / CRF** (3 h 30 → 2 h) | §5–7 CRF, apprentissage, Viterbi ; §9 évaluation stricte / partielle ; §10 Transformer + CRF | §2 schémas BIO/BIOES | §3 corpus de mots, §8 spans imbriqués |
| **m13 Multi-tâche** (3 h 30 → 2 h) | §3 gradients, §5 méthodes d'équilibrage, §7 mesurer le transfert, §9 protocole d'ablation | §4 Pareto | §6 expérience jouet (si le temps manque) |

**Livrable de l'étape** : une page « protocole » pour ProtAnnotML : tâches T1–T4, régimes de découpage, métriques par résidu / domaine / borne, ablations prévues (DAPT, TAPT, pondération des pertes, tête régions, LoRA vs complet), nombre de graines. C'est la future section « Méthodes ».

+++

## Étape 3 — Le fil rouge (≈ 13 h)

| Notebook | Durée | Ce qui change avec tes vraies données |
|---|---|---|
| **pf01 Données IMGT** | 2 h 30 | remplacer le générateur par les fichiers IMGT/GENE-DB et les sorties AIRR d'IMGT/HighV-QUEST ; mesurer ambiguïtés et taux de rejet |
| **pf02 Splits par homologie** | 1 h 30 | MMseqs2 sur les séquences, clustering des germlines ; ajouter le split temporel sur IMGT/mAb-DB |
| **pf03 Références** | 2 h 30 | comparer aussi avec IMGT/DomainGapAlign et ANARCI sur les mêmes séquences |
| **pf04 Modèle** | 3 h | **lancer `smoke_test.py` en premier** ; puis 35M sur le GPU de 6 Go, 650M sur V100 |
| **pf05 Évaluation** | 2 h | IC par cluster, tests appariés, risque–couverture du gène |
| **pf06 Passage à l'échelle** | 1 h 30 | scripts Slurm Jean Zay (remplacer `CHANGE_ME`), DAPT sur H100 |

**Livrable de l'étape** : un premier tableau de résultats R1/R2/R3 sur les vraies données, modèle 35M ou 150M contre alignement, avec intervalles de confiance.

+++

## Étape 4 — Comprendre et publier (≈ 8 h)

| Activité | Durée | Livrable |
|---|---|---|
| **m19 Interprétabilité** | 3 h | sondes par couche, gradients intégrés sur les frontières CDR, contacts C23–C104 depuis l'attention |
| **pf05 revisité** sur le modèle 650M | 2 h | figures stratifiées (identité, mutation, locus, espèce), analyse d'erreurs à la main |
| **Lectures** 1, 3, 5, 6, 9, 10, 12, 15 | 3 h et plus | fiches de 5 lignes → section « travaux connexes » |

+++

## Optionnel A — Structure, contraste, distillation (≈ 8 h)

- **m20 Protéines et structure** (3 h) : si la délimitation des domaines dans les constructions (T1) reste le point faible ; ablation SaProt.
- **m11 Apprentissage contrastif** (3 h, survoler §2–3 et faire §8–9) : pour une identification de gène par **récupération** (plus proches germlines dans l'espace des plongements), qui gère naturellement les gènes ajoutés au répertoire sans réentraînement.
- **m12 Distillation** (2 h 30) : pour servir un petit modèle (8M–35M) rapide sur le site IMGT, distillé du 650M.

+++

## Optionnel B — Assistant IMGT, KG-RAG et agents (≈ 20 h)

Pour ton travail sur IMGT-KG, indépendamment de ProtAnnotML :

1. **m15 Agents, outils et MCP** (4 h) : appel d'outils, sécurité, serveurs MCP.
2. **m16 RAG** (4 h, survol possible des §2–3 sur l'ingestion PDF) : recherche documentaire sur la documentation IMGT.
3. **m17 KG-RAG** (4 h) : RDF, SPARQL, Fuseki, text-to-SPARQL validé, GraphRAG.
4. **m21 KG-RAG sur IMGT/mAb-KG** (4 h) : application directe à ton graphe.
5. **m18 Multi-agents** (3 h 30) : seulement si un agent unique ne suffit pas (lire la section 9 d'abord).
6. **m14 Affiner un LLM** (4 h) : seulement si tu veux affiner un modèle de text-to-SPARQL sur tes propres paires question / requête.

+++

## Conseils d'organisation

- **Alterne théorie et projet** dès l'étape 2 : fais pf01 sur tes vraies données pendant que tu lis m08 ; tu poseras de meilleures questions.
- **Garde un journal d'expériences** (date, configuration, graine, résultat, conclusion en une ligne) : c'est la matière première de l'article et de la réponse aux relecteurs.
- **Ne passe pas au 650M** avant que le 35M ne batte la régression logistique du fil rouge 3 sur les mêmes données : si le petit modèle ne fonctionne pas, le gros ne résoudra pas le problème.
- **Décide tôt de la question de l'article** : vu les résultats du fil rouge 3, la contribution la plus défendable n'est probablement pas « une meilleure numérotation des domaines V standard », mais l'identification avec abstention, la robustesse aux espèces peu couvertes et aux constructions, et la vitesse.
