# Principaux concepts – « Attention Is All You Need »

**Article fondateur du Transformer (Vaswani et al., 2017, arXiv:1706.03762)**

## Idées clés

- Le Transformer est une architecture de traitement de séquence qui repose uniquement sur des mécanismes d’attention (plus de RNN ni de CNN)[web:7][web:9].
- L’attention permet de modéliser efficacement les dépendances globales d’une séquence : chaque élément peut interagir directement avec tous les autres, quelle que soit leur distance dans la séquence.
- Innovation majeure : l’auto-attention multi-têtes (multi-head self-attention) et une structure modulaire en piles de couches encoder/decoder.
- Les positions ne sont pas apprises par la structure même du réseau (pas de temps comme dans les RNN) : on ajoute donc des encodages positionnels pour injecter l’ordre des tokens.
- Avantages : parallélisable, rapide à entraîner, plus efficace sur les dépendances longues et interprétable (visualisation des têtes d’attention).
- Le Transformer surpasse largement les modèles précédents en traduction automatique (et d’autres tâches comme le parsing) avec moins de ressources de calcul.
- Utilisation d’Adam, du dropout et du label smoothing à l’entraînement.

> Pour une citation directe : « We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. »[web:7]

---

ℹ️ Pour aller plus loin : [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) (Vaswani et al.)
