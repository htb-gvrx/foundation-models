## 1. Contexte

Avant 2017, les meilleurs modèles de traduction et de séquences sont basés sur :

- **RNN / LSTM / GRU** : séquentiels, difficiles à paralléliser  
- **CNN (ByteNet, ConvS2S)** : réduisent la profondeur séquentielle mais utilisent des kernels locaux et des chemins longs pour relier des tokens éloignés  

Les auteurs partent du constat suivant :

> Le goulot d’étranglement principal, ce n’est plus le nombre de paramètres,  
> mais la part de calcul **non parallélisable** (la dépendance séquentielle).

---

## 2. Contribution principale : le Transformer

Ils proposent une architecture de traduction **sans aucune récursivité ni convolution**, basée uniquement sur :

### 2.1 Self-attention multi-tête (multi-head self-attention)

- **Scaled dot-product attention** :

  $$
  \text{Attention}(Q,K,V)
  = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
  $$
Q (Query) : "requête"

Il s’agit d’un vecteur (ou matrice) représentant ce que l’on cherche à récupérer ou comparer.

Pour chaque position analysée dans la séquence, un vecteur “query” est produit.

K (Key) : "clé"

Le vecteur "key" sert à représenter comment chaque élément/mot peut être “retrouvé” ou “adressé”.

Chaque mot/élément de la séquence dispose d’un vecteur clé.

V (Value) : "valeur"

Le vecteur "value" transporte le contenu à extraire si la clé correspond à une requête.

Chaque élément a également un vecteur valeur.

dₖ : "dimension des clés" (souvent noté 
d
k
d 
k
 )

C’est la dimension/longueur des vecteurs “key” (et “query”).

Le terme 
d
k
d 
k
 
  dans le dénominateur sert à normaliser le produit scalaire des requêtes et des clés pour éviter des valeurs trop grandes, ce qui stabilise l’apprentissage.

Résumé :

Q, K, V sont des matrices/vecteurs produits à partir des entrées (par multiplication par des poids appris),

dₖ est la dimension des “keys” (et “queries”), utilisée pour la normalisation.

Chaque position de la séquence finale dans le modèle “attend” différents endroits du contexte selon le score d’attention calculé par cette équation.


- Plusieurs « têtes » d’attention en parallèle pour regarder différents sous-espaces de représentation.

### 2.2 Architecture encodeur–décodeur empilée (N = 6 couches)

Chaque couche comprend :

- Multi-head self-attention  
- Réseau feed-forward positionnel (FFN) appliqué à chaque position  
- Connexions résiduelles + LayerNorm partout  

Côté décodeur :

- Self-attention **masqué** (pour ne pas voir le futur)  
- Attention sur la sortie de l’encodeur.

### 2.3 Encodage positionnel sinusoïdal

- Pas de RNN → il faut injecter l’ordre.  
- Ajout au vecteur d’embedding de sinusoïdes de fréquences différentes, permettant de représenter des positions absolues et relatives.

### 2.4 Apprentissage & régularisation

- Optimiseur Adam avec schedule de learning rate spécifique  
  (warmup + décroissance en \(1 / \sqrt{\text{step}}\))  
- Dropout sur les sous-couches et les embeddings  
- Label smoothing pour réduire la sur-confiance des sorties  

---

## 3. Résultats

Sur WMT 2014 :

- **EN→DE** : BLEU 28.4 (big Transformer) – plus de 2 BLEU de mieux que tous les modèles précédents, y compris les ensembles (ensembles de modèles).  
- **EN→FR** : BLEU 41.8 – nouvel état de l’art pour un modèle unique, avec un coût d’entraînement très inférieur aux concurrents.

Le modèle :

- est plus rapide à entraîner (bien parallélisable sur GPU),  
- obtient de meilleures performances,  
- se généralise aussi bien à une tâche différente : **parsing en constituants** (Penn Treebank WSJ), où il rivalise voire dépasse des modèles spécialisés.  

---

## 4. Pourquoi “self-attention” ? (analyse interne de l’article)

Les auteurs comparent trois familles de couches :

### 4.1 RNN

- Complexité : \(O(n \cdot d^2)\)  
- Séquentiel : \(O(n)\) opérations dépendantes  
- Chemin max entre deux positions : \(O(n)\)

### 4.2 CNN

- Complexité : \(O(k \cdot n \cdot d^2)\) (ou moins avec convolutions séparables)  
- Chemin max : \(O(n/k)\) ou \(O(\log_k n)\) avec dilatations

### 4.3 Self-attention

- Complexité : \(O(n^2 \cdot d)\)  
- Séquentiel : \(O(1)\) (tout est parallélisable sur la séquence)  
- Chemin max : \(O(1)\) – chaque position voit toutes les autres en une seule couche.

**Argument clé :**

> En réduisant la longueur de chemin entre deux positions à une constante,  
> on facilite l’apprentissage des dépendances longues.

---

## 5. Forces de l’article

### 5.1 Un design extrêmement cohérent

- **Architecture minimaliste** :  
  tout repose sur trois briques : attention, MLP positionnel, normalisation + résidu.
- **Encodeur / décodeur symétriques**, partage de beaucoup d’idées (mêmes tailles, mêmes patterns).

Pour un architecte, c’est séduisant : la complexité conceptuelle est faible, la complexité computationnelle est assumée, et tout se ramène à des multiplications de matrices optimisées.

### 5.2 Argument de performance complet

L’article ne se contente pas d’un « nouveau bloc de réseau » :

- Benchmarks sérieux sur des datasets standardisés et difficiles (WMT14, WSJ).  
- Comparaison explicite avec les coûts de FLOPs des autres modèles (GNMT, ConvS2S, MoE, etc.).  
- Études d’ablation : variations sur :
  - le nombre de têtes,  
  - la taille de \(d_k / d_v\),  
  - la taille des couches (\(d_{\text{model}}, d_{\text{ff}}\)),  
  - le taux de dropout,  
  - l’encodage positionnel (sinusoïdal vs learned).

Ce n’est pas parfait, mais c’est bien plus rigoureux qu’un simple « on a battu le SOTA ».

### 5.3 Intuition interprétable

Les visualisations d’attention (en annexe) montrent :

- Des têtes qui suivent des dépendances syntaxiques (verbe–complément, anaphores « its → Law »…)  
- Des têtes spécialisées dans certains motifs (délimitation de segments, accords, etc.).

Cela renforce l’argument : **self-attention est un bon biais inductif pour la langue**.

---

## 6. Limites et angles morts

### 6.1 Complexité quadratique en longueur de séquence

Le papier revendique que :

> La self-attention est plus efficace que les RNN/CNN,  
> dans le régime des phrases (longueur de phrase \(\ll d\)).

Mais :

- La complexité \(O(n^2)\) en mémoire et en calcul devient vite problématique dès que l’on dépasse quelques centaines ou milliers de tokens.  
- L’article mentionne la possibilité d’une self-attention restreinte à un voisinage de taille \(r\), mais ne l’explore pas vraiment.

**Critique :**  
Pour les cas *long context / long documents / code / logs / multimodal*, le papier ne donne pas de solution opérationnelle ; c’est une dette technique qui sera payée plus tard (Transformers efficaces, sparse attention, etc.).

### 6.2 Domaine et métriques limités

Les résultats sont excellents sur :

- Traduction EN–DE, EN–FR  
- Parsing en constituants

Mais :

- Pas d’évaluation systématique sur la **robustesse** (bruit, out-of-domain, langues basses ressources).  
- Pas de métriques de **calibration, biais, alignement** – ce n’est pas l’objet de l’article, mais aujourd’hui ce sont des dimensions critiques.  
- La « généralisation à d’autres tâches » est démontrée par un seul exemple (parsing), avec peu de tuning.

### 6.3 Slogan un peu trompeur : “Attention is all you need”

En pratique, l’article utilise quand même :

- des MLP profonds par position (\(d_{\text{ff}} = 2048\)),  
- des encodages positionnels non triviaux,  
- une régularisation sophistiquée,  
- un schedule de learning rate spécifique.

Le vrai message technique est plutôt :

> Pour des tâches de séquence type traduction,  
> une stack de couches **attention + MLP** suffit, sans RNN ni CNN.

La formule « is all you need » est brillante pour le marketing scientifique, mais occulte le rôle crucial :

- du **scale** (données, compute),  
- de la conception de pipeline (BPE, batching, beam search, etc.).

### 6.4 Interprétabilité : promesse plus que solution

Les figures d’attention sont séduisantes, mais :

- on voit que certaines têtes suivent des dépendances linguistiques,  
- l’article ne démontre pas que l’attention est globalement interprétable ni fiable comme outil d’explication.

C’est une ouverture intéressante, pas encore une solution d’XAI.
