# ♔ Mouvement du Roi — Calcul Matriciel sur GPU

## Introduction

### Pourquoi simuler les échecs avec des matrices ?

Aux échecs, chaque coup demande de répondre à une question simple : **quelles cases une pièce
peut-elle atteindre ?** Habituellement, un programme parcourt les cases une par une avec des
boucles et des conditions (`if` la case est libre, `if` on ne sort pas du plateau…). Ça
fonctionne, mais c'est **séquentiel** — on traite les cases l'une après l'autre.

L'idée de ce document est de reformuler cette question comme un **calcul matriciel**. Au lieu
de boucler sur les 64 cases, on construit une matrice 8×8 dont chaque cellule vaut 1 (case
atteignable) ou 0 (case non atteignable). Ce calcul se fait **en une seule opération**
mathématique, sans boucle, et peut donc être exécuté massivement en parallèle sur un GPU.

### Pourquoi le Roi comme premier exemple ?

Le Roi est la pièce la plus simple à modéliser : il se déplace d'**une seule case** dans
n'importe quelle direction (horizontale, verticale ou diagonale). Contrairement à la Dame
ou à la Tour, il n'a pas de mouvement « glissant » qui nécessite de détecter les obstacles
le long d'une ligne. Son masque de mouvement est donc **fixe et local** — un carré 3×3
centré sur sa position — ce qui en fait le candidat idéal pour comprendre les bases du
calcul matriciel appliqué aux échecs, avant de passer aux pièces plus complexes.

### Ce que nous allons montrer

1. Comment **encoder un échiquier** dans une matrice de nombres entiers
2. Comment **localiser le Roi** avec une matrice binaire de position
3. Comment calculer les **cases atteignables** avec une seule formule mathématique (la distance de Tchebychev)
4. Comment cette formule se traduit en **opération GPU** (convolution 2D / shifts matriciels)
5. Comment **paralléliser** le calcul pour simuler des milliers de parties en même temps

---

## 1. Représentation de l'échiquier

Un échiquier standard comporte 8 rangées et 8 colonnes, soit 64 cases. On le représente
naturellement comme une matrice $B$ de 8 lignes et 8 colonnes, dont chaque coefficient est
un entier qui identifie la pièce présente (ou 0 si la case est vide).

$$
B \in \mathbb{Z}^{8 \times 8}
$$

$$
B = \begin{pmatrix}
b_{0,0} & b_{0,1} & \cdots & b_{0,7} \\
b_{1,0} & b_{1,1} & \cdots & b_{1,7} \\
\vdots  & \vdots  & \ddots & \vdots  \\
b_{7,0} & b_{7,1} & \cdots & b_{7,7}
\end{pmatrix}
$$

Chaque coefficient $b_{i,j}$ prend une valeur entière selon la convention suivante :

| Valeur | Pièce blanche | | Valeur | Pièce noire |
|--------|---------------|-|--------|-------------|
| $+1$   | Pion ♙        | | $-1$   | Pion ♟      |
| $+2$   | Cavalier ♘    | | $-2$   | Cavalier ♞  |
| $+3$   | Fou ♗         | | $-3$   | Fou ♝       |
| $+4$   | Tour ♖        | | $-4$   | Tour ♜      |
| $+5$   | Dame ♕        | | $-5$   | Dame ♛      |
| $+6$   | Roi ♔         | | $-6$   | Roi ♚       |
| $0$    | Case vide     | |        |             |

> **Convention d'indexation** : la ligne $i = 0$ correspond à la rangée 8
> (côté noir) et la ligne $i = 7$ à la rangée 1 (côté blanc). La colonne
> $j = 0$ correspond à la colonne `a` et $j = 7$ à la colonne `h`.

Pour la suite de ce document, nous travaillons sur un **échiquier vide** avec uniquement
le Roi blanc, afin d'isoler et de comprendre le calcul de son mouvement.

---

## 2. Position du Roi

Avant de calculer les cases atteignables, il faut dire **où se trouve le Roi**
dans la matrice. On pourrait simplement stocker ses coordonnées $(r, c)$, mais
pour rester dans le formalisme matriciel (et préparer la version GPU), on
encode sa position sous forme d'une **matrice binaire** $P$ de même taille que
l'échiquier : toutes les cases valent 0 sauf celle du Roi qui vaut 1.

Le Roi est placé en $(r, c)$. On définit sa **matrice de position** $P \in \{0, 1\}^{8 \times 8}$ :

$$
P_{i,j} = \begin{cases} 1 & \text{si } (i, j) = (r, c) \\ 0 & \text{sinon} \end{cases}
$$

---

## 3. Distance de Tchebychev

### L'intuition

Comment déterminer si une case $(i, j)$ est atteignable par le Roi en un seul coup ?
Le Roi peut se déplacer d'**une case** dans chacune des 8 directions : haut, bas, gauche,
droite, et les 4 diagonales. Autrement dit, depuis $(r, c)$, il peut atteindre toute case
$(i, j)$ telle que la distance en lignes **et** la distance en colonnes soient **toutes
deux** inférieures ou égales à 1.

Cette notion correspond exactement à une distance mathématique bien connue :
la **distance de Tchebychev** (aussi appelée distance de l'échiquier, ou $L_\infty$).

### Définition

On construit la **matrice de distances** $D \in \mathbb{N}^{8 \times 8}$ par la distance de Tchebychev :

$$
D_{i,j} = \max\!\Big(\left|i - r\right|,\; \left|j - c\right|\Big)
$$

> La distance de Tchebychev correspond exactement au nombre minimum de
> coups qu'un Roi met pour atteindre une case sur un échiquier vide.
> C'est la raison pour laquelle elle est parfois appelée « distance du Roi ».

En comparaison avec d'autres distances :
- La **distance de Manhattan** ($L_1$) additionne les écarts : $|i - r| + |j - c|$. Elle
  correspondrait à une pièce qui ne peut se déplacer qu'en horizontal/vertical (pas en diagonale).
- La **distance euclidienne** ($L_2$) donne $\sqrt{(i-r)^2 + (j-c)^2}$. Elle n'est pas
  adaptée à un plateau discret en cases.
- La **distance de Tchebychev** ($L_\infty$) prend le maximum des écarts. C'est la bonne
  métrique car le Roi peut couvrir un écart en ligne **et** en colonne simultanément
  grâce aux diagonales.

---

## 4. Masque de mouvement

### De la distance au masque binaire

On a maintenant une matrice $D$ qui donne, pour chaque case de l'échiquier, la distance
(en nombre de coups du Roi) par rapport à sa position. Pour obtenir les cases atteignables
**en un seul coup**, il suffit de ne garder que les cases où $D = 1$.

On construit un **masque binaire** $M$ : une matrice 8×8 de 0 et de 1, où 1 signifie
« le Roi peut aller sur cette case ».

Les cases atteignables en **un seul coup** sont celles à distance exactement 1 :

$$
\boxed{\; M_{i,j} = \mathbb{1}\!\left[\; D_{i,j} = 1 \;\right] \;}
$$

soit, explicitement :

$$
M_{i,j} = \begin{cases} 1 & \text{si } \max\!\left(\left|i - r\right|,\; \left|j - c\right|\right) = 1 \\ 0 & \text{sinon} \end{cases}
$$

> **Point clé** : cette formule est le cœur de tout le document. Elle ne contient
> aucune boucle, aucun `if`, aucune condition sur les bords. C'est une opération
> appliquée à **chaque cellule indépendamment** — exactement le type de calcul
> qu'un GPU exécute en parallèle sur des milliers de cœurs.

---

## 5. Exemple concret — Roi en $(4, 3)$

Prenons un exemple concret pour vérifier que la formule fonctionne. On place le Roi
blanc en $(4, 3)$, ce qui correspond à la case **d4** en notation échiquéenne standard.

### Étape 1 : construire la matrice de distances $D$

On applique la formule $D_{i,j} = \max(|i - 4|, |j - 3|)$ à chacune des 64 cases.
Par exemple :
- Case $(4, 3)$ (le Roi lui-même) : $\max(|4-4|, |3-3|) = \max(0, 0) = 0$
- Case $(3, 2)$ (diagonale haut-gauche) : $\max(|3-4|, |2-3|) = \max(1, 1) = 1$ ✓ atteignable
- Case $(2, 1)$ (deux cases en diagonale) : $\max(|2-4|, |1-3|) = \max(2, 2) = 2$ ✗ trop loin
- Case $(4, 0)$ (même ligne, 3 cases à gauche) : $\max(|4-4|, |0-3|) = \max(0, 3) = 3$ ✗ trop loin

### Matrice de distances $D$

$$
D_{i,j} = \max(|i - 4|,\; |j - 3|)
$$

$$
D = \begin{pmatrix}
4 & 4 & 4 & 4 & 4 & 4 & 4 & 4 \\
3 & 3 & 3 & 3 & 3 & 3 & 3 & 4 \\
3 & 2 & 2 & 2 & 2 & 2 & 3 & 4 \\
3 & 2 & 1 & 1 & 1 & 2 & 3 & 4 \\
3 & 2 & 1 & 0 & 1 & 2 & 3 & 4 \\
3 & 2 & 1 & 1 & 1 & 2 & 3 & 4 \\
3 & 2 & 2 & 2 & 2 & 2 & 3 & 4 \\
3 & 3 & 3 & 3 & 3 & 3 & 3 & 4
\end{pmatrix}
$$

### Étape 2 : extraire le masque $M = \mathbb{1}[D = 1]$

On ne garde que les cases où $D$ vaut exactement 1. Toutes les autres deviennent 0 :

$$
M = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

On retrouve bien les **8 cases** adjacentes au Roi, formant un carré 3×3 avec un
trou au centre (la position du Roi lui-même).

Visualisation sur l'échiquier :

```
    a   b   c   d   e   f   g   h
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
8 │   │   │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
7 │   │   │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
6 │   │   │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
5 │   │   │ × │ × │ × │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
4 │   │   │ × │ ♔ │ × │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
3 │   │   │ × │ × │ × │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
2 │   │   │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
1 │   │   │   │   │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┘
```

---

## 6. Cas en bord et en coin — Roi en $(0, 0)$

Un avantage majeur de la formule par distance de Tchebychev est qu'elle **gère
automatiquement les bords** de l'échiquier. On n'a pas besoin d'ajouter des conditions
du type « si la case existe ». Les indices $(i, j)$ vont de 0 à 7 par définition de
la matrice, donc les cases en dehors du plateau n'apparaissent tout simplement jamais
dans le calcul.

Vérifions avec le cas extrême : le Roi est dans le **coin** en $(0, 0)$ (case a8).

$$
D = \begin{pmatrix}
0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
1 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
2 & 2 & 2 & 3 & 4 & 5 & 6 & 7 \\
\vdots & & & & & & & \vdots \\
7 & 7 & 7 & 7 & 7 & 7 & 7 & 7
\end{pmatrix}
\quad \Longrightarrow \quad
M = \begin{pmatrix}
0 & 1 & 0 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 0 \\
0 & 0 & 0 & \cdots & 0 \\
\vdots & & & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 0
\end{pmatrix}
$$

> Le bord de l'échiquier **clip naturellement** — pas besoin de condition
> supplémentaire. La formule $\max(|i|, |j|) = 1$ ne produit que **3 cases**
> au lieu de 8. En bord (mais pas dans un coin), on obtiendrait 5 cases.
> La formule s'adapte toute seule, sans cas particulier à gérer.

---

## 7. Formulation par shifts (convolution) — version GPU

### Pourquoi une autre formulation ?

La formule par distance de Tchebychev est élégante et correcte, mais pour l'implémenter
efficacement sur GPU, il est plus naturel de la reformuler comme une **convolution 2D**.
Les GPU sont en effet optimisés pour les convolutions (c'est l'opération de base du
deep learning), avec des bibliothèques comme cuDNN qui les exécutent en quelques
microsecondes.

### L'idée : décaler et sommer

Au lieu de calculer une distance, on procède autrement :
1. On part de la matrice de position $P$ (un seul 1, tout le reste à 0)
2. On la **décale** (shift) dans chacune des 8 directions
3. On **additionne** les 8 matrices décalées

Chaque décalage produit un 1 sur une case adjacente. La somme des 8 décalages donne
exactement le masque de mouvement $M$.

Le masque $M$ peut aussi être obtenu par **somme de 8 décalages** de la matrice de position $P$ :

$$
M = \sum_{(d_r,\, d_c) \;\in\; \mathcal{D}} \mathrm{shift}(P,\; d_r,\; d_c)
$$

où $\mathcal{D}$ est l'ensemble des 8 directions :

$$
\mathcal{D} = \Big\{(-1,-1),\; (-1,0),\; (-1,+1),\; (0,-1),\; (0,+1),\; (+1,-1),\; (+1,0),\; (+1,+1)\Big\}
$$

et $\mathrm{shift}(P, d_r, d_c)$ décale la matrice $P$ de $(d_r, d_c)$ en remplissant les
bords avec des zéros (les cases qui « sortent » du plateau disparaissent, et les cases
qui « entrent » sont vides — exactement le comportement souhaité pour les bords) :

$$
\mathrm{shift}(P, d_r, d_c)_{i,j} = \begin{cases}
P_{i - d_r,\; j - d_c} & \text{si } 0 \le i - d_r < 8 \text{ et } 0 \le j - d_c < 8 \\
0 & \text{sinon}
\end{cases}
$$

### Illustration d'un shift

Par exemple, $\mathrm{shift}(P, +1, 0)$ décale la matrice d'une case vers le bas.
Si le Roi est en $(4, 3)$, le 1 se retrouve en $(5, 3)$ — la case juste en dessous.
En sommant les 8 shifts, on obtient un 1 sur chacune des 8 cases adjacentes.

### Équivalence avec une convolution 2D

> **Équivalence avec convolution 2D** : cette opération est strictement
> équivalente à une convolution avec le noyau $3 \times 3$ :
>
> $$
> K = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \end{pmatrix}
> $$
>
> $$
> M = P * K \quad \text{(convolution 2D, padding = 1, stride = 1)}
> $$
>
> Ce noyau $K$ encode les 8 directions du Roi : chaque 1 dans $K$ correspond
> à une direction de mouvement. Le 0 au centre signifie que le Roi ne
> « reste pas sur place ». Le padding de 1 assure que la sortie a la même
> taille que l'entrée (8×8), et les bords sont naturellement gérés par le
> remplissage à zéro du padding.

---

## 8. Parallélisme GPU — batch de $N$ parties

### Le vrai pouvoir du GPU

Calculer le masque de mouvement d'un seul Roi sur un seul plateau est trivial — même
un CPU le fait instantanément. L'intérêt du calcul matriciel et du GPU apparaît quand
on veut **simuler des milliers de parties en parallèle**, par exemple pour explorer un
arbre de jeu (tous les coups possibles, puis tous les coups suivants, etc.).

Au lieu de traiter chaque partie séquentiellement, on empile $N$ plateaux dans un
**tenseur 3D** de dimension $(N, 8, 8)$ et on applique la même opération à tous
simultanément. Le GPU exécute alors $N \times 64$ calculs en parallèle grâce à ses
milliers de cœurs CUDA.

Pour $N$ parties simultanées, on travaille sur un tenseur $\mathbf{P} \in \{0,1\}^{N \times 8 \times 8}$ :

$$
\mathbf{M} = \sum_{(d_r, d_c) \in \mathcal{D}} \mathrm{shift}(\mathbf{P},\; d_r,\; d_c)
$$

L'opération est **identique** mais appliquée sur la dimension batch — le GPU traite les $N$ plateaux en parallèle en une seule instruction. Concrètement, en PyTorch, cela revient à :

```python
import torch.nn.functional as F

# K : noyau 3×3 du Roi, P : positions des Rois (N, 1, 8, 8)
M = F.conv2d(P, K, padding=1)
# → M a la forme (N, 1, 8, 8) : les N masques calculés en un seul appel GPU
```

**Complexité** :

| | CPU (séquentiel) | GPU (parallèle) |
|---|---|---|
| 1 plateau | $O(64)$ | $O(1)$ |
| $N$ plateaux | $O(64N)$ | $O(1)$ |

> Le $O(1)$ GPU est en termes de *temps-horloge* grâce au parallélisme
> massif — chaque cellule $(n, i, j)$ est calculée par un thread CUDA distinct.

---

## 9. Résumé

| Concept | Formule | Rôle |
|---|---|---|
| Échiquier | $B \in \mathbb{Z}^{8 \times 8}$ | Encoder l'état du jeu |
| Position du Roi | $P_{i,j} = \mathbb{1}[(i,j) = (r,c)]$ | Localiser la pièce |
| Distance | $D_{i,j} = \max(\|i-r\|, \|j-c\|)$ | Mesurer l'éloignement (Tchebychev) |
| Masque de mouvement | $M_{i,j} = \mathbb{1}[D_{i,j} = 1]$ | Cases atteignables en 1 coup |
| Version GPU | $M = P * K$ (convolution 2D) | Parallélisation massive |

> **Prochaine étape** : les pièces glissantes (Tour, Fou, Dame) nécessitent
> un mécanisme de **propagation avec blocage** — le calcul est plus riche
> mais repose sur les mêmes principes de shifts matriciels.
