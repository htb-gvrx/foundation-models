# Softmax

La fonction **softmax** est une fonction mathématique utilisée pour transformer un vecteur de valeurs réelles en une distribution de probabilités.

## Définition mathématique

Soit un vecteur de \( n \) valeurs réelles \((x_1, x_2, ..., x_n)\) :  
La sortie de la softmax est un vecteur \((s_1, s_2, ..., s_n)\) où, pour chaque composante \( i \) :

\[
s_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
\]

- Chaque valeur \( s_i \) est comprise entre 0 et 1, et la somme de toutes les \( s_i \) est égale à 1.
- On obtient ainsi une **distribution de probabilités** adaptée pour des tâches de classification (par exemple, prédire quelle classe est la plus probable).

## Utilisation typique

- En sortie d'un réseau de neurones pour la classification multi-classe.
- Permet d'interpréter les résultats comme des probabilités.

## Exemple :

Si on applique softmax à \([2, 1, 0]\) :

\[
s_1 = \frac{e^{2}}{e^{2} + e^{1} + e^{0}} \\
s_2 = \frac{e^{1}}{e^{2} + e^{1} + e^{0}} \\
s_3 = \frac{e^{0}}{e^{2} + e^{1} + e^{0}}
\]

Cela donne un vecteur où la valeur la plus élevée correspond à la case la plus "probable" selon le modèle.

---

**Résumé :**  
La fonction softmax sert à convertir des scores arbitraires en probabilités, particulièrement utile dans les modèles de classification automatique.
