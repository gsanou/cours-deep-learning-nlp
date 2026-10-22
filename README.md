# Deep Learning pour le NLP : de la rétropropagation aux modèles de langage de protéines

Trente et un notebooks Jupyter : le chapitre 0 refait les maths à la main (avec animations et un labo interactif) ; les modules 0 à 9 forment le parcours principal, les modules 10 à 13 approfondissent la NER, l'apprentissage contrastif, la distillation et le multi-tâche, et les modules 14 à 18 (partie III) couvrent les applications LLM : affinage, agents et MCP, RAG, KG-RAG avec Jena Fuseki, multi-agents avec LangGraph. La partie IV est centrée sur **ProtAnnotML** : un parcours raccourci, un projet fil rouge en six notebooks (données IMGT, découpages par homologie, références par alignement, modèle PLM multi-tâche, évaluation statistique, passage à l'échelle sur Jean Zay), les modules 19 (interprétabilité), 20 (structure) et 21 (KG-RAG sur IMGT/mAb-KG), et une liste de lecture commentée. Le dossier `projet_protannotml/` contient le paquet Python, les scripts, les configurations et les scripts Slurm exportés du fil rouge. Le fichier `demo_kg.ttl` est le graphe RDF fictif du module 17. Chaque section suit le même schéma : **intuition → maths → code → visualisation**, avec exercices corrigés en fin de module.

| # | Notebook | Contenu | Durée |
|---|---|---|---|
| ∂ | `c00_maths_a_la_main.ipynb` | chapitre 0, à la main : produit scalaire, matrices, perceptron, dérivées, règle de la chaîne, gradient, softmax et entropie croisée, MLP 2-2-1 calculé entièrement (avant, arrière, mise à jour), 6 animations ; labo interactif `site/labo_chapitre0.html` | 3 h 30 |
| 0 | `m00_fondements.ipynb` | neurone, activations, pertes (MLE), descente de gradient, Momentum/RMSProp/Adam, rétropropagation (mini-autograd), MLP NumPy, initialisation, régularisation, passage à PyTorch | 3–4 h |
| 1 | `m01_pytorch.ipynb` | tenseurs, vues, broadcasting, einsum, autograd (VJP), `nn.Module`, `DataLoader` + padding/masques, boucle robuste, débogage | 2–3 h |
| 2 | `m02_texte_vers_nombres.ipynb` | Unicode, Zipf, BPE de zéro, vocabulaire, TF-IDF, classifieur de sentiment, métriques, tokenizers HF | 2 h 30 |
| 3 | `m03_embeddings.ipynb` | PPMI + SVD, word2vec SGNS (gradients dérivés, NumPy), analogies, `nn.Embedding`, biais | 2 h 30 |
| 4 | `m04_rnn.ipynb` | n-grammes, perplexité, char-RNN NumPy (BPTT), gradient évanescent, LSTM/GRU, échantillonnage | 3 h |
| 5 | `m05_seq2seq_attention.ipynb` | encodeur-décodeur, attention de Bahdanau/Luong, traducteur de dates avec alignements | 2 h 30 |
| 6 | `m06_transformer.ipynb` | √d_k, masques, multi-têtes, RoPE, LayerNorm/résiduels, GPT NumPy, mini-GPT PyTorch | 4 h |
| 7 | `m07_pretraining_hf.ipynb` | CLM/MLM, lois d'échelle, Hugging Face, affinage, étiquetage de tokens (BIO, F1 par entité) | 3 h |
| 8 | `m08_adaptation.ipynb` | DAPT/TAPT, LoRA, multi-tâche (incertitude homoscédastique), fuite par similarité, bootstrap, calibration | 3 h 30 |
| 9 | `m09_protein_lm.ipynb` | co-évolution, régions FR/CDR + Viterbi, ESM-2 (embeddings, affinage, LoRA, mutations zéro-shot) | 3 h |
| 10 | `m10_ner.ipynb` | BIO/BIOES, CRF linéaire de zéro (forward, forward-backward, Viterbi), spans imbriqués, évaluation stricte/partielle, Transformer + CRF | 3 h 30 |
| 11 | `m11_contrastive.ipynb` | InfoNCE, température, alignement/uniformité, SimCSE, sentence-transformers, négatifs difficiles, biologie | 3 h |
| 12 | `m12_distillation.ipynb` | cibles douces, facteur T², connaissance sombre, TinyBERT/DistilBERT, distillation d'un outil existant | 2 h 30 |
| 13 | `m13_multitask.ipynb` | Pareto, conflits de gradients, incertitude/GradNorm/DWA/PCGrad, Δ_MTL, affinité entre tâches, MMoE | 3 h 30 |
| 14 | `m14_llm_finetuning.ipynb` | gabarits de chat, masque de perte, packing, mémoire LoRA/QLoRA, SFT et DPO (dérivation + démo), TRL, GRPO, évaluation | 4 h |
| 15 | `m15_agents_tools_mcp.ipynb` | appel d'outils (JSON Schema), boucle d'agent, recherche web et documentaire, injection de prompt, serveur et client MCP | 4 h |
| 16 | `m16_rag.ipynb` | ingestion PDF, découpage, BM25, dense, hybride (RRF), Recall@k/MRR/nDCG, citations, fidélité, LangChain/FAISS | 4 h |
| 17 | `m17_kg_rag.ipynb` | RDF/RDFS/OWL, SPARQL, inférence, Jena Fuseki, text-to-SPARQL validé, GraphRAG, serveur MCP SPARQL | 4 h |
| 18 | `m18_multiagents_langgraph.ipynb` | graphes d'état, routeur, superviseur, relecture, map-reduce, checkpoints, humain dans la boucle, LangChain/LangGraph, frameworks | 3 h 30 |

### Partie IV · ProtAnnotML

| # | Notebook | Contenu | Durée |
|---|---|---|---|
| → | `parcours_protannotml.ipynb` | parcours raccourci (~40 h) : quoi faire, survoler, sauter, dans quel ordre | lecture |
| F1 | `pf01_donnees_imgt.ipynb` | tâches T1–T4, numérotation IMGT (gaps, insertions), FASTA IMGT/GENE-DB, AIRR d'IMGT/HighV-QUEST, données synthétiques, ambiguïtés d'allèles | 2 h 30 |
| F2 | `pf02_splits_homologie.ipynb` | identité sur alignements IMGT, clusters, régimes R0–R3, fuite résiduelle, MMseqs2, split temporel | 1 h 30 |
| F3 | `pf03_baselines.ipynb` | numérotation par alignement (à la DomainGapAlign), modèle par résidu, Viterbi monotone et BIO, tests de résistance | 2 h 30 |
| F4 | `pf04_modele_plm.ipynb` | ESM-2 multi-tâche, CRF contraint, cibles ensemblistes, DAPT, LoRA, précision mixte, reprise, test rapide | 3 h |
| F5 | `pf05_evaluation.ipynb` | bootstrap par cluster, tests appariés (McNemar), stratification, risque–couverture, liste de contrôle d'article | 2 h |
| F6 | `pf06_echelle_deploiement.ipynb` | budgets mémoire et temps (6 Go, V100, H100), Slurm Jean Zay, reproductibilité, CLI AIRR, service HTTP, fiche de modèle | 2 h |
| 19 | `m19_interpretabilite.ipynb` | sondes et tâches de contrôle, gradients intégrés, occlusion, tests de validité, SAE TopK, ablations, recettes ESM-2 | 3 h |
| 20 | `m20_proteines_structure.ipynb` | repliement Ig et numérotation IMGT, contacts, GNN, délimitation spectrale, 3Di/SaProt, ESMFold, ImmuneBuilder, ESM3 | 3 h |
| 21 | `m21_kg_rag_imgt_mabkg.ipynb` | introspection du schéma, liaison d'entités, text-to-SPARQL évalué, GraphRAG, plongements et fuite par inverses, MCP, LangGraph | 4 h |
| ≡ | `lectures_commentees.ipynb` | 16 articles clés (Transformer, ESM, AMPLIFY, AntiBERTy, AbLang, LoRA, DPO, RAG…) : à retenir, limites | 20–30 h |

## Exécution

```bash
pip install numpy matplotlib scikit-learn scipy torch transformers datasets accelerate peft sentence-transformers
jupyter lab
```

Ou ouvre chaque notebook dans Google Colab (GPU conseillé à partir du module 5).

## À savoir

- Les cellules **NumPy** (implémentations de zéro et visualisations) ont été exécutées : leurs sorties et figures sont dans les notebooks.
- Les cellules **PyTorch / Hugging Face** n'ont pas pu être exécutées dans l'environnement de construction (pas d'accès à PyTorch). Elles ont été vérifiées statiquement (syntaxe, noms) et sont écrites pour tourner telles quelles ; leurs sorties apparaîtront à ta première exécution. Si une cellule casse (version de librairie différente, identifiant du Hub déplacé), le commentaire à côté indique en général l'alternative.
- Les jeux de données des parties exécutées sont soit réels et livrés avec Python (`pydoc_data`), soit **synthétiques** et présentés comme tels.

## Reconstruire

`src/*.md` contient les sources ; `python tools/build.py src/m00_fondements.md` régénère et réexécute un notebook ; `python tools/make_site.py` régénère la version HTML. Les cellules marquées `export=…` dans les sources du fil rouge sont aussi écrites dans `projet_protannotml/` : reconstruire les notebooks pf01 à pf06 régénère le paquet. Exécute les notebooks du fil rouge dans l'ordre (pf01 crée les données utilisées par les suivants, dans `notebooks/data_protannotml/`).

## Déployer avec Git (GitHub Pages)

Le dépôt est prêt : `.gitignore` (les données générées et les archives ne sont pas versionnées) et un workflow `.github/workflows/pages.yml` qui publie le dossier `site/` à chaque `push` sur `main`.

1. Crée un dépôt vide sur GitHub (sans README), par exemple `cours-deep-learning-nlp`.
2. Dans ce dossier :
   ```bash
   git remote add origin git@github.com:<ton-compte>/cours-deep-learning-nlp.git
   git push -u origin main
   ```
3. Sur GitHub : *Settings → Pages → Build and deployment → Source : GitHub Actions*. Le site est ensuite en ligne à `https://<ton-compte>.github.io/cours-deep-learning-nlp/` (labo du chapitre 0 : `labo.html`).

Pour mettre à jour : modifie `src/*.md`, reconstruis (`python tools/build.py src/<module>.md`, puis `python tools/make_site.py` et `python tools/wrap_page.py site/labo_chapitre0.html site/labo.html`), puis `git commit` et `git push`.

GitHub Pages est gratuit pour un dépôt **public** ; pour un dépôt privé, il faut un compte payant (Pro, Team ou Enterprise). Les notebooks s'ouvrent aussi directement dans Colab depuis GitHub (`https://colab.research.google.com/github/<ton-compte>/<dépôt>/blob/main/notebooks/<notebook>.ipynb`), ce qui permet d'exécuter les cellules PyTorch sur GPU.

