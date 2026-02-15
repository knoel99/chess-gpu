# ♟ Prédiction de coups aux échecs — Modèle linéaire matriciel

## Introduction

### Le projet

On dispose de **millions de parties d'échecs** jouées sur Chess.com, accessibles via
leur API publique. Chaque partie est une séquence de positions et de coups. L'objectif
est d'entraîner un modèle capable de **prédire le prochain coup** à partir de la position
actuelle du plateau.

### Pourquoi commencer par un modèle linéaire ?

Dans un réseau de neurones classique, on empile des couches de transformations avec des
fonctions d'activation non-linéaires (ReLU, sigmoid…). Chaque couche ajoutée permet au
modèle de capturer des relations plus complexes entre les entrées. Mais cette puissance
a un coût : plus de paramètres, plus de calcul, plus de difficulté à comprendre ce que
le modèle a appris.

Nous faisons ici le choix inverse : **le modèle le plus simple possible**, une seule
multiplication matricielle, sans couche cachée, sans framework (pas de PyTorch, pas de
TensorFlow). Juste NumPy et des matrices.

Ce choix est délibéré pour plusieurs raisons :

1. **Comprendre avant d'optimiser** — un modèle qu'on peut écrire en 20 lignes de code
   est un modèle qu'on comprend entièrement. Chaque poids a une interprétation directe.

2. **Valider le pipeline** — avant de lancer un entraînement GPU de plusieurs heures,
   on veut s'assurer que la tokenisation, le chargement des données et la boucle
   d'entraînement fonctionnent correctement.

3. **Établir un baseline** — ce modèle simple donne un score de référence. Tout modèle
   plus complexe devra faire mieux, sinon sa complexité n'est pas justifiée.

4. **Tout est du calcul matriciel** — le forward pass, le calcul de la loss, le gradient
   et la mise à jour des poids sont tous exprimables comme des opérations matricielles.
   C'est exactement ce qu'on veut pour exploiter un GPU plus tard.

---

## 1. Formulation du problème

### Ce qu'on veut prédire

À chaque instant d'une partie, il y a :
- une **position** : quelles pièces sont sur quelles cases
- un **coup joué** : le coup que le joueur a effectivement choisi

On veut apprendre une fonction :

$$f(\text{position}) \longrightarrow \text{coup}$$

Plus précisément, on veut une **distribution de probabilité** sur tous les coups possibles.
Le coup le plus probable selon le modèle est sa prédiction.

### Données d'entraînement

Chaque partie de $n$ coups fournit $n$ exemples d'entraînement :

```
Partie : 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 ...

Exemple 1 : position initiale         → coup = e4
Exemple 2 : position après 1.e4       → coup = e5
Exemple 3 : position après 1.e4 e5    → coup = Nf3
Exemple 4 : position après 1...Nc6    → coup = Bb5
...
```

Une partie de 40 coups donne **80 exemples** (40 coups blancs + 40 coups noirs).
Avec 364 parties de Magnus en janvier 2025, on a déjà ~30 000 exemples.

---

## 2. Tokenisation de l'entrée — encoder le plateau

### Pourquoi ne pas utiliser les entiers directement ?

L'échiquier est une matrice 8×8. On pourrait l'aplatir en un vecteur de 64 entiers
dans $[-6, +6]$ :

$$x_{\text{brut}} = [0, 0, 0, 5, 0, 0, -4, 0, \ldots] \in \mathbb{Z}^{64}$$

Mais un modèle linéaire interprète ces valeurs **numériquement**. Or :
- Dame = 5, Tour = 4 → numériquement proches, stratégiquement très différentes
- Pion blanc = +1, Pion noir = -1 → le modèle croirait qu'un pion noir est
  « le négatif » d'un pion blanc, ce qui n'a aucun sens tactique

Il faut un encodage qui traite chaque type de pièce comme une **catégorie distincte**.

### Encodage one-hot par case

Chaque case de l'échiquier est dans l'un de **13 états** :

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-------|---|---|---|---|---|---|---|---|---|---|----|----|-----|
| Pièce | vide | ♙ | ♘ | ♗ | ♖ | ♕ | ♔ | ♟ | ♞ | ♝ | ♜ | ♛ | ♚ |

Une case contenant une Dame blanche (♕) est encodée :

$$[\underbrace{0}_{vide}, \underbrace{0}_{♙}, \underbrace{0}_{♘}, \underbrace{0}_{♗}, \underbrace{0}_{♖}, \underbrace{1}_{♕}, \underbrace{0}_{♔}, \underbrace{0}_{♟}, \underbrace{0}_{♞}, \underbrace{0}_{♝}, \underbrace{0}_{♜}, \underbrace{0}_{♛}, \underbrace{0}_{♚}]$$

On concatène les 64 cases :

$$x = [\underbrace{x_{a8}}_{\text{13 bits}}, \underbrace{x_{b8}}_{\text{13 bits}}, \ldots, \underbrace{x_{h1}}_{\text{13 bits}}] \in \{0, 1\}^{832}$$

Ce vecteur est **creux** (sparse) : sur 832 entrées, exactement 64 valent 1 (une par case)
et le reste vaut 0.

### Propriétés de cet encodage

- **Pas d'ordre artificiel** : le modèle ne suppose pas que Dame > Tour > Fou
- **Chaque pièce a son propre axe** : le poids associé à "Dame blanche en d1" est
  totalement indépendant du poids associé à "Tour blanche en d1"
- **Compatible GPU** : le vecteur one-hot est une ligne d'une matrice creuse,
  et le produit $W \cdot x$ revient à sélectionner des colonnes de $W$

---

## 3. Tokenisation de la sortie — encoder les coups

### Qu'est-ce qu'un coup ?

Un coup aux échecs est défini par :
- une **case de départ** (64 possibilités)
- une **case d'arrivée** (64 possibilités)
- éventuellement une **promotion** (Dame, Tour, Fou, Cavalier)

### Dictionnaire de coups

On énumère **tous les coups géométriquement possibles** sur un échiquier 8×8 et on
attribue un index unique à chacun :

| Type de coup | Exemples | Nombre |
|---|---|---|
| Glissements horizontaux/verticaux | Tour/Dame sur lignes et colonnes | 896 |
| Glissements diagonaux | Fou/Dame sur diagonales | 560 |
| Sauts de cavalier | 8 directions depuis chaque case | 336 |
| Avances de pion (1 et 2 cases) | Pions blancs et noirs | 112 |
| Captures de pion en diagonale | Captures et en passant | ~48 |
| Promotions | ×4 types (Q, R, B, N) par direction | ~96 |
| **Total** | | **1 968 coups** |

> Certains de ces coups ne sont jamais légaux dans une position donnée
> (par exemple un pion qui recule). Ce n'est pas grave : le modèle apprendra
> simplement à leur donner une probabilité proche de 0.

Chaque coup est un **token**. Le modèle produit un vecteur de 1 968 probabilités,
une par coup possible.

---

## 4. Le modèle — une seule matrice

### Formule

$$\boxed{p = \text{softmax}(W \cdot x + b)}$$

où :

| Symbole | Dimension | Description |
|---|---|---|
| $x$ | $(832, 1)$ | Position du plateau (one-hot aplati) |
| $W$ | $(1968, 832)$ | Matrice de poids — **c'est le modèle** |
| $b$ | $(1968, 1)$ | Vecteur de biais |
| $W \cdot x + b$ | $(1968, 1)$ | Scores bruts (logits) pour chaque coup |
| $p$ | $(1968, 1)$ | Probabilités après softmax |

### Softmax — transformer des scores en probabilités

Les logits $z = W \cdot x + b$ sont des nombres réels quelconques (positifs, négatifs,
grands, petits). La fonction softmax les transforme en probabilités :

$$p_k = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{1968} e^{z_j}}$$

Propriétés :
- Chaque $p_k \in (0, 1)$
- $\sum_k p_k = 1$ (c'est une distribution de probabilité)
- Le coup avec le plus grand logit a la plus grande probabilité
- L'exponentielle **amplifie les écarts** : un logit légèrement plus grand
  donne une probabilité significativement plus grande

### Prédiction

Le coup prédit est celui de probabilité maximale :

$$\hat{y} = \underset{k}{\text{argmax}} \; p_k$$

### Taille du modèle

$$\text{paramètres} = \underbrace{1968 \times 832}_{W} + \underbrace{1968}_{b} = 1\,639\,344$$

En float32, cela occupe **6.3 Mo** en mémoire — un modèle minuscule.

---

## 5. Interprétation des poids

Un avantage majeur du modèle linéaire est la **lisibilité des poids**. Chaque coefficient
de $W$ a une signification directe :

$$W[\text{coup}_{k},\; \text{case}_{i} \cdot 13 + \text{pièce}_{j}]$$

= contribution de « pièce $j$ sur case $i$ » au score du coup $k$.

### Exemple

Après entraînement, on s'attend à trouver :

```
W[coup "Cf3", case g1, pièce ♘]  =  forte valeur positive
  → "Un Cavalier blanc en g1 rend le coup Cg1-f3 probable"

W[coup "e4",  case e2, pièce ♙]  =  forte valeur positive
  → "Un pion blanc en e2 rend le coup e2-e4 probable"

W[coup "e4",  case e2, pièce vide] = forte valeur négative
  → "Si e2 est vide, on ne peut pas jouer e2-e4"
```

On peut **visualiser** la matrice $W$ comme une carte de chaleur pour comprendre
ce que le modèle a appris.

---

## 6. Entraînement — descente de gradient

### Fonction de perte : cross-entropy

Pour un exemple d'entraînement $(x, y)$ où $y$ est le coup réellement joué, la
perte (loss) mesure à quel point le modèle se trompe :

$$\mathcal{L} = -\log(p_{y})$$

où $p_y$ est la probabilité que le modèle assigne au coup correct.

Intuition :
- Si $p_y = 0.9$ (le modèle est confiant et correct) → $\mathcal{L} = -\log(0.9) = 0.105$ (petite perte)
- Si $p_y = 0.01$ (le modèle se trompe) → $\mathcal{L} = -\log(0.01) = 4.6$ (grande perte)

### Gradient — la direction pour s'améliorer

Le gradient de $\mathcal{L}$ par rapport aux poids a une forme remarquablement simple.
On définit le vecteur d'erreur :

$$e = p - y_{\text{one-hot}}$$

où $y_{\text{one-hot}}$ est le vecteur de taille 1 968 avec un 1 à l'index du coup
joué et des 0 partout ailleurs. Alors :

$$\frac{\partial \mathcal{L}}{\partial W} = e \cdot x^\top \qquad \frac{\partial \mathcal{L}}{\partial b} = e$$

> C'est un **produit extérieur** entre le vecteur d'erreur $e$ (1968,) et le
> vecteur d'entrée $x$ (832,). Le résultat est une matrice (1968 × 832) de
> même dimension que $W$.

### Mise à jour des poids

$$W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}$$
$$b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}$$

où $\eta$ est le taux d'apprentissage (learning rate), typiquement $\eta \approx 0.01$.

### Tout en une seule passe

```
x   (832,)     donnée ─────────────────────────────────────────────┐
                                                                    │
                    ┌──────────────┐                                │
                    │  z = W·x + b │  (produit matrice-vecteur)     │
                    └──────┬───────┘                                │
                           │ (1968,)                                │
                    ┌──────▼───────┐                                │
                    │  p = softmax │  (exponentielle + normalisation)│
                    └──────┬───────┘                                │
                           │ (1968,)                                │
                    ┌──────▼───────┐                                │
                    │ e = p - y    │  (soustraction)                │
                    └──────┬───────┘                                │
                           │ (1968,)                                │
                    ┌──────▼───────┐                   ┌────────────▼─┐
                    │ ∂W = e · xᵀ  │◄──────────────────│   x (832,)   │
                    └──────┬───────┘  produit extérieur └──────────────┘
                           │ (1968, 832)
                    ┌──────▼───────┐
                    │ W ← W - η·∂W │  (mise à jour)
                    └──────────────┘
```

**Chaque opération est un calcul matriciel standard** : produit matrice-vecteur,
exponentielle élément-par-élément, soustraction, produit extérieur. Aucune boucle
sur les neurones, aucune bibliothèque spécialisée nécessaire.

---

## 7. En batch — plusieurs exemples à la fois

En pratique, on ne traite pas les exemples un par un. On regroupe $N$ exemples en un
**batch** et on fait tous les calculs en une seule opération matricielle :

$$X \in \{0,1\}^{N \times 832} \qquad Z = X \cdot W^\top + b^\top \qquad P = \text{softmax}(Z)$$

| Symbole | Dimension | Description |
|---|---|---|
| $X$ | $(N, 832)$ | $N$ positions empilées |
| $W^\top$ | $(832, 1968)$ | Transposée de $W$ |
| $Z$ | $(N, 1968)$ | Logits pour tous les exemples |
| $P$ | $(N, 1968)$ | Probabilités pour tous les exemples |

Le gradient devient :

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} \cdot E^\top \cdot X$$

où $E = P - Y$ est la matrice d'erreur $(N, 1968)$.

C'est un **unique produit de matrices** — exactement ce pour quoi les GPU sont conçus.

---

## 8. Ce que ce modèle ne peut pas apprendre

### La limite fondamentale

Le score de chaque coup est une **somme** de contributions indépendantes :

$$\text{score}(\text{coup}_k) = \sum_{i=1}^{832} W_{k,i} \cdot x_i = \sum_{\text{case } c} \sum_{\text{pièce } p} W_{k,(c,p)} \cdot \mathbb{1}[\text{pièce } p \text{ sur case } c]$$

Il n'y a **aucun terme croisé** $x_i \cdot x_j$. Chaque case contribue au score
indépendamment de toutes les autres. Le modèle ne peut donc pas exprimer :

> « **SI** le Fou est en b2 **ET** le Cavalier est en f6 **ET** le Roi est en g7,
> **ALORS** le Cavalier est cloué et ne peut pas bouger »

Ce type de raisonnement nécessite une **interaction multiplicative** entre features,
ce qui demande au minimum une couche cachée avec non-linéarité.

### Concrètement, ce qui échappe au modèle

| Concept | Pourquoi c'est impossible pour un modèle linéaire |
|---|---|
| **Clouages** | Nécessite de croiser 3 positions (attaquant, pièce clouée, roi) |
| **Fourchettes** | Le coup est bon parce que 2 pièces adverses sont aux bonnes cases |
| **Batteries** | La force vient de l'alignement de 2 pièces alliées |
| **Sacrifices** | Perdre du matériel pour un gain positionnel = relation non-linéaire |
| **Structure de pions** | Les pions interagissent entre eux (chaînes, îlots) |

### Performances attendues

| Modèle | Précision top-1 estimée | Précision top-5 estimée |
|---|---|---|
| **Aléatoire** | ~0.05% (1/1968) | ~0.25% |
| **Ce modèle (linéaire)** | ~5-10% | ~20-30% |
| 1 couche cachée (256 neurones) | ~20-25% | ~50% |
| Réseau profond (type Leela) | ~55% | ~85% |

Un score de 5-10% peut sembler faible, mais c'est **100× mieux que le hasard** — le
modèle a bien capturé les patterns statistiques de base (ouvertures courantes,
mouvements typiques par pièce, biais positionnel).

---

## 9. Résumé de l'architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ENTRÉE                                                        │
│   Position 8×8 (13 types de pièces)                             │
│                        │                                        │
│                        ▼                                        │
│                 ┌──────────────┐                                │
│                 │  One-hot     │                                │
│                 │  64 × 13     │                                │
│                 │  = 832 bits  │                                │
│                 └──────┬───────┘                                │
│                        │  x (832,)                              │
│                        ▼                                        │
│     ┌──────────────────────────────────────┐                    │
│     │        z = W · x + b                 │                    │
│     │                                      │                    │
│     │   W : 1968 × 832  (1.6M paramètres)  │                    │
│     │   b : 1968         (biais)           │                    │
│     │                                      │                    │
│     │   Une seule multiplication matricielle│                   │
│     └──────────────────┬───────────────────┘                    │
│                        │  z (1968,)                             │
│                        ▼                                        │
│                 ┌──────────────┐                                │
│                 │   softmax    │                                │
│                 └──────┬───────┘                                │
│                        │  p (1968,)                             │
│                        ▼                                        │
│   SORTIE                                                        │
│   Probabilité de chaque coup (1968 tokens)                      │
│   Prédiction = argmax(p)                                        │
│                                                                 │
│   Entraînement : cross-entropy + descente de gradient           │
│   ∂W = (p − y) · xᵀ      ← un seul produit extérieur          │
│   W ← W − η · ∂W         ← une seule soustraction matricielle  │
│                                                                 │
│   Paramètres : 1 639 344  │  Mémoire : 6.3 Mo (float32)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
