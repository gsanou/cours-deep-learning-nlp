# Liste de lecture commentée — les articles qui comptent pour ProtAnnotML

> Seize articles, dans l'ordre où je te conseille de les lire, avec pour chacun : pourquoi le lire, ce qu'il faut en retenir, ses **limites** (lecture de relecteur), et le lien avec ProtAnnotML et les modules du cours. Puis quelques compléments ciblés. Compte une à deux heures par article pour une lecture active (figures, méthodes, tableaux de résultats), moins pour ceux que tu connais déjà.

| | |
|---|---|
| **Durée estimée** | 20 à 30 h de lecture, à étaler en parallèle du fil rouge |
| **Méthode conseillée** | pour chaque article, écrire 5 lignes : question, méthode, résultat principal, limite, ce que j'en fais |

> 📌 Les liens pointent vers les versions librement accessibles (arXiv, bioRxiv, PMC) quand elles existent. Pour les préprints, vérifie s'il existe une version publiée plus récente avant de citer.

+++

## A. Les fondations (architecture et pré-entraînement)

### 1. Vaswani et al. (2017) — *Attention Is All You Need* (NeurIPS)
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · modules 5–6

- **Retenir** : l'auto-attention remplace la récurrence ; attention multi-têtes, connexions résiduelles, normalisation, encodage de position ; tout est parallélisable sur la séquence.
- **Limites** : coût quadratique en longueur (sans gravité pour des domaines V de ~120 résidus, mais réel pour des chaînes entières ou des constructions) ; l'encodage de position sinusoïdal a été remplacé dans les PLM récents par des positions **rotatives** (RoPE, utilisées par ESM-2).
- **Pour ProtAnnotML** : comprendre pourquoi un Transformer voit la frontière V → C qu'une fenêtre locale manque (fil rouge 3) : chaque résidu peut regarder le motif J à plus de 10 positions.

### 2. Devlin et al. (2019) — *BERT* (NAACL)
[arXiv:1810.04805](https://arxiv.org/abs/1810.04805) · module 7

- **Retenir** : pré-entraînement par **modèle de langage masqué** (MLM, 15 % des tokens) puis affinage avec une petite tête par tâche ; c'est exactement le schéma d'ESM-2 et de ProtAnnotML.
- **Limites** : la tâche NSP s'est révélée inutile (RoBERTa) ; seuls 15 % des tokens fournissent un signal d'apprentissage par passe ; les hyperparamètres d'affinage sont instables sur petits jeux (plusieurs graines nécessaires).
- **Pour ProtAnnotML** : la DAPT est un « BERT continué » ; le choix du taux de masquage et du corpus compte plus que les détails d'architecture.

### 3. Gururangan et al. (2020) — *Don't Stop Pretraining* (ACL)
[arXiv:2004.10964](https://arxiv.org/abs/2004.10964) · module 8, fil rouge 4

- **Retenir** : continuer le pré-entraînement sur le **domaine** (DAPT) puis sur les données de la **tâche** (TAPT) améliore l'affinage, surtout quand le domaine est éloigné du corpus initial ; les deux se cumulent.
- **Limites** : démontré en NLP (biomédical, informatique, actualités, avis) ; le gain dépend de la distance au domaine et diminue avec beaucoup de données étiquetées ; coût de calcul non négligeable.
- **Pour ProtAnnotML** : c'est la justification de ta DAPT sur IG/TR. À démontrer par **ablation** (avec / sans DAPT, avec / sans TAPT), pas à supposer.

+++

## B. Les modèles de langage de protéines

### 4. Rives et al. (2021) — *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences* (PNAS)
[doi:10.1073/pnas.2016239118](https://doi.org/10.1073/pnas.2016239118) · module 9

- **Retenir** : un Transformer entraîné en MLM sur des séquences seules (ESM-1b) apprend des propriétés biochimiques, la structure secondaire, des contacts, des effets de mutations, sans supervision ; ces propriétés sont **décodables par sondes linéaires**.
- **Limites** : les sondes montrent ce qui est décodable, pas ce qui est utilisé (module 19) ; les évaluations dépendent fortement du découpage par homologie.
- **Pour ProtAnnotML** : la base conceptuelle « la séquence suffit à apprendre beaucoup de structure » ; méthodologie de sondes à reprendre pour l'article.

### 5. Lin et al. (2023) — *Evolutionary-scale prediction of atomic-level protein structure with a language model* (Science)
[doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) · modules 9, 19, 20

- **Retenir** : la famille **ESM-2** (8M → 15B paramètres), la structure qui émerge avec l'échelle (contacts, puis ESMFold qui prédit la 3D à partir d'une seule séquence, sans alignement multiple).
- **Limites** : les gains au-delà de quelques centaines de millions de paramètres sont modestes pour beaucoup de tâches d'affinage ; corpus UniRef avec ses biais d'échantillonnage (article 15) ; ESMFold est moins précis qu'AlphaFold2 avec alignement multiple, en particulier sur les boucles variables.
- **Pour ProtAnnotML** : ton tronc par défaut (35M pour itérer, 650M pour les résultats). Mesure le gain 150M → 650M → 3B **avant** de payer le 3B.

### 6. Fournier et al. (2024) — *Protein Language Models: Is Scaling Necessary?* (AMPLIFY, bioRxiv)
[doi:10.1101/2024.09.23.614603](https://doi.org/10.1101/2024.09.23.614603) · fil rouge 4 et 6

- **Retenir** : deux modèles de 120M et 350M paramètres, présentés comme compétitifs avec des PLM bien plus grands pour une fraction du coût, grâce à une architecture moderne, un code efficace et surtout l'attention portée à la **qualité des données** (y compris le biais d'échantillonnage) ; code, données et poids publiés (licence Apache 2.0).
- **Limites** : préprint (vérifie le statut de publication) ; les comparaisons dépendent des jeux de validation choisis et de la façon de mesurer la perplexité ; les évaluations en aval sont moins nombreuses que pour ESM-2.
- **Pour ProtAnnotML** : argument fort pour investir dans la **curation** de ton corpus de DAPT plutôt que dans la taille ; candidat de tronc alternatif (exercice du fil rouge 4).

### 7. Ruffolo et al. (2021) — *Deciphering antibody affinity maturation with language models and weakly supervised learning* (AntiBERTy, NeurIPS MLSB workshop)
[arXiv:2112.07782](https://arxiv.org/abs/2112.07782) · module 9

- **Retenir** : un modèle de langage spécifique aux anticorps, entraîné sur 558 millions de séquences d'OAS ; utilisation d'un apprentissage à instances multiples pour identifier des résidus liés à la maturation ; ses plongements servent ensuite à IgFold (prédiction de structure d'anticorps).
- **Limites** : article d'atelier, évaluation limitée ; corpus dominé par des répertoires peu mutés (article 9) ; la redondance d'OAS impose un regroupement soigneux.
- **Pour ProtAnnotML** : point de comparaison « modèle spécialisé anticorps » face à ESM-2 + DAPT. Pour un article, au moins un modèle spécialisé doit figurer dans les comparaisons.

### 8. Olsen, Moal & Deane (2022) — *AbLang: an antibody language model for completing antibody sequences* (Bioinformatics Advances)
[doi:10.1093/bioadv/vbac046](https://doi.org/10.1093/bioadv/vbac046) · module 9

- **Retenir** : modèles séparés pour chaînes lourdes et légères entraînés sur OAS ; application utile et concrète : **restaurer les résidus manquants** (notamment en N-terminal, souvent tronqué par le séquençage).
- **Limites** : chaînes non appariées ; même biais de germline que les autres modèles d'anticorps (article 9).
- **Pour ProtAnnotML** : les séquences tronquées en N-terminal sont fréquentes ; ton modèle doit les numéroter correctement (le fil rouge en simule 15 %).

### 9. Olsen, Moal & Deane (2024) — *Addressing the antibody germline bias and its effect on language models for improved antibody design* (AbLang-2)
[bioRxiv doi:10.1101/2024.02.02.578678](https://doi.org/10.1101/2024.02.02.578678) · fil rouge 1 et 4

- **Retenir** : les modèles d'anticorps prédisent la **germline** pour la grande majorité des résidus non germinaux masqués (environ 85–88 % pour Sapiens, AntiBERTy, AbLang-1 ; environ 50 % pour ESM-2), parce que les données d'entraînement viennent surtout de cellules B naïves ou non triées. La perplexité globale masque ce défaut. AbLang-2 utilise une **perte focale** et un affinage sur paires VH/VL pour mieux prédire les résidus mutés.
- **Limites** : l'étiquette « non germinal » dépend des bases de germlines utilisées par IgBLAST/ANARCI ; un allèle manquant transforme un résidu germinal en faux « mutant ».
- **Pour ProtAnnotML** : double enseignement. (1) Mesure séparément les performances sur résidus germinaux et mutés. (2) La limite citée est exactement là où l'expertise IMGT pèse : la qualité du répertoire de référence conditionne la qualité des étiquettes.

### 10. Schmirler et al. (2024) — *Fine-tuning protein language models boosts predictions across diverse tasks* (Nature Communications)
[PMC11358375](https://pmc.ncbi.nlm.nih.gov/articles/PMC11358375) · module 8, fil rouge 4

- **Retenir** : comparaison systématique entre plongements gelés et **affinage** (y compris LoRA) de plusieurs PLM sur des tâches au niveau de la protéine et du résidu ; l'affinage améliore la plupart des tâches, et LoRA donne l'essentiel du gain pour une fraction du coût.
- **Limites** : le gain varie selon la tâche et le modèle ; sur de petits jeux, risque de surapprentissage et forte variance entre graines.
- **Pour ProtAnnotML** : justifie le choix « affinage (complet ou LoRA) » plutôt que « plongements gelés + tête ». Montre-le par ablation sur T1–T4.

+++

## C. Adapter, combiner, aligner

### 11. Hu et al. (2022) — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR)
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685) · modules 8 et 14

- **Retenir** : on gèle les poids et on apprend une mise à jour de **faible rang** $\Delta W = BA$ ; très peu de paramètres entraînables, pas de surcoût à l'inférence une fois fusionnée.
- **Limites** : le rang et les modules ciblés sont des hyperparamètres ; pour un changement de domaine important, l'affinage complet peut rester meilleur ; la mémoire des **activations** n'est pas réduite.
- **Pour ProtAnnotML** : la configuration V100 du fil rouge (650M + LoRA) ; compare LoRA et affinage complet sur H100.

### 12. Kendall, Gal & Cipolla (2018) — *Multi-Task Learning Using Uncertainty to Weigh Losses* (CVPR), avec Xin et al. (2022) — *Do Current Multi-Task Optimization Methods in Deep Learning Even Help?* (NeurIPS)
[arXiv:1705.07115](https://arxiv.org/abs/1705.07115) · [arXiv:2209.11379](https://arxiv.org/abs/2209.11379) · modules 8 et 13

- **Retenir** : Kendall pondère chaque perte par une incertitude apprise ($e^{-s_k} L_k + s_k$), ce qui évite de régler les poids à la main. Xin et al. montrent qu'à réglage d'hyperparamètres égal, de nombreuses méthodes d'optimisation multi-tâche ne battent pas une simple **somme pondérée** bien réglée.
- **Limites** : la formulation de Kendall suppose des vraisemblances gaussiennes ou softmax ; les poids appris peuvent diverger si une perte devient très petite.
- **Pour ProtAnnotML** : garde la pondération par incertitude, mais **compare-la** à des poids égaux et à une grille simple, sur plusieurs graines (fil rouge 4, exercice 2). C'est la question que posera un relecteur informé.

### 13. Rafailov et al. (2023) — *Direct Preference Optimization* (NeurIPS)
[arXiv:2305.18290](https://arxiv.org/abs/2305.18290) · module 14

- **Retenir** : on aligne un modèle sur des préférences (réponse A préférée à B) **sans** modèle de récompense ni apprentissage par renforcement, via une perte de classification sur des paires ; β contrôle l'écart au modèle de référence.
- **Limites** : sensible à la qualité et à la couverture des paires ; tendance au surapprentissage des préférences ; pas directement utile pour l'annotation.
- **Pour ProtAnnotML** : hors du cœur du projet ; utile si tu affines un LLM pour l'interface (assistant d'annotation, agent IMGT) ou, plus tard, pour la conception d'anticorps guidée par des préférences expérimentales.

### 14. Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS)
[arXiv:2005.11401](https://arxiv.org/abs/2005.11401) · modules 16, 17, 21

- **Retenir** : un générateur conditionné par des passages récupérés (mémoire non paramétrique) : les connaissances peuvent être mises à jour sans réentraîner, et les réponses citées.
- **Limites** : les erreurs de récupération se propagent ; la fidélité aux sources doit être mesurée, pas supposée ; la version originale entraîne le récupérateur et le générateur ensemble, ce que les systèmes actuels font rarement.
- **Pour ProtAnnotML** : la base de l'assistant IMGT : interroger IMGT/mAb-KG (module 21) et la documentation IMGT avec citations. Aussi une idée pour T4 : une **numérotation par récupération** (plus proches germlines dans l'espace des plongements, module 11).

+++

## D. Structure, numérotation et biais

### 15. Ding & Steinhardt (2024) — *Protein language models are biased by unequal sequence sampling across the tree of life* (bioRxiv)
[doi:10.1101/2024.03.07.584001](https://doi.org/10.1101/2024.03.07.584001) · fil rouge 2 et 5

- **Retenir** : les scores (vraisemblances) des PLM sont biaisés par la représentation inégale des espèces dans les bases de séquences : les espèces sur-représentées obtiennent de meilleurs scores, indépendamment de la fonction.
- **Limites** : analyse surtout sur les vraisemblances ; l'effet sur les tâches affinées est moins documenté.
- **Pour ProtAnnotML** : justifie le régime R3 (espèce exclue) et la stratification par espèce ; c'est aussi un argument pour la valeur d'IMGT, qui couvre des espèces peu présentes dans UniRef.

### 16. Su et al. (2024) — *SaProt: Protein Language Modeling with Structure-aware Vocabulary* (ICLR)
[bioRxiv doi:10.1101/2023.10.01.560349](https://doi.org/10.1101/2023.10.01.560349) · module 20

- **Retenir** : chaque token combine un acide aminé et une lettre structurale 3Di (Foldseek) ; avec des structures prédites (AlphaFold DB), le modèle surpasse les PLM de séquence sur de nombreuses tâches.
- **Limites** : dépend de la qualité des structures prédites (les régions peu fiables sont masquées) ; coût d'une prédiction de structure par séquence ; gain variable selon les tâches.
- **Pour ProtAnnotML** : ablation intéressante pour T1 (constructions sans linker) ; pas indispensable pour T2–T3 standard.

+++

## E. Compléments ciblés

| Article | Pourquoi | Module |
|---|---|---|
| Lefranc et al. (2003), *IMGT unique numbering for immunoglobulin and T cell receptor variable domains and Ig superfamily V-like domains*, Dev. Comp. Immunol. [doi:10.1016/S0145-305X(02)00039-3](https://doi.org/10.1016/S0145-305X(02)00039-3) | la définition de tes étiquettes ; à citer et à résumer pour les relecteurs ML | fil rouge 1 |
| Dunbar & Deane (2016), *ANARCI: antigen receptor numbering and receptor classification*, Bioinformatics [doi:10.1093/bioinformatics/btv552](https://doi.org/10.1093/bioinformatics/btv552) | l'autre référence de numérotation (HMM de profils) à inclure dans les comparaisons | fil rouge 3 |
| Hewitt & Liang (2019), *Designing and Interpreting Probes with Control Tasks*, EMNLP [arXiv:1909.03368](https://arxiv.org/abs/1909.03368) | sondes honnêtes | module 19 |
| Adebayo et al. (2018), *Sanity Checks for Saliency Maps*, NeurIPS [arXiv:1810.03292](https://arxiv.org/abs/1810.03292) | tester les attributions | module 19 |
| Simon & Zou (2024), *InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders* [arXiv:2412.12101](https://arxiv.org/abs/2412.12101) | SAE sur ESM-2 | module 19 |
| Kenlay et al. (2024), IgBert / IgT5 (modèles d'anticorps appariés) | modèles d'anticorps récents sur paires VH/VL | module 9 |

+++

## Méthode de lecture : la fiche de 5 lignes

Pour chaque article, avant de passer au suivant :

1. **Question** : quel problème, et pourquoi maintenant ?
2. **Méthode** : en une phrase, ce qui est nouveau.
3. **Résultat** : le chiffre ou la figure qui porte l'article, avec le jeu de test et le découpage utilisés.
4. **Limite** : ce qu'un relecteur exigeant contesterait (fuite, baselines faibles, une seule graine, métrique choisie a posteriori…).
5. **Pour moi** : ce que j'en reprends, ce que je teste, ce que je cite.

Les fiches deviennent directement la section « travaux connexes » de ton article, et la ligne 4 alimente ta propre liste de contrôle (fil rouge 5).
