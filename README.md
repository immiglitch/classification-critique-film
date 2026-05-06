# TP final — Classification de critiques de films avec scikit-learn

## 1. Présentation du projet

Dans ce dossier, vous trouverez :

- `TP_final-squelette.py` : le script principal du projet final ;
- `TP8_TP9_IDL.ipynb` : le notebook dans lequel j’ai réalisé les exercices d’entraînement ;
- `imdb_smol.tar.gz` : l’archive contenant les critiques de films ;
- `wine.data` : fichier utilisé dans les exercices du TP ;
- `README.md` : le fichier d’explication du projet.

Ce projet a pour objectif de réaliser un système de classification automatique de critiques de films à partir du corpus `imdb_smol`.

L’archive fournie contient 602 critiques de films sous forme de fichiers textes. Ces critiques sont réparties en deux classes :

- `pos` : critiques positives ;
- `neg` : critiques négatives.

---

## 2. Chargement des données

Le chargement est réalisé dans la méthode `get_data()` de la classe `entrainement_modele`.

Cette méthode renvoie deux listes :

```python
textes, y
```

où :

- `textes` contient les critiques de films sous forme de chaînes de caractères ;
- `y` contient les labels associés, c’est-à-dire `pos` ou `neg`.

Exemple de résultat obtenu :

```text
Nombre de textes chargés : 602
Nombre de labels chargés : 602
Labels trouvés : {'pos', 'neg'}
```

---

## 3. Vectorisation des textes

Les modèles de machine learning ne peuvent pas traiter directement du texte brut. Il faut donc transformer les critiques en vecteurs numériques.

Dans la version actuelle du projet, la vectorisation utilisée est :

```python
TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    sublinear_tf=True
)
```

Le choix de `TfidfVectorizer` est pertinent pour une tâche de classification de documents, car il ne se contente pas de compter les mots. Il pondère chaque mot selon son importance dans un document et dans l’ensemble du corpus.

Cela permet de donner plus de poids aux mots discriminants comme :

```text
excellent, boring, awful, amazing, terrible, disappointing
```

et de réduire l’importance des mots trop fréquents et peu informatifs.

### Paramètres utilisés

- `stop_words="english"` : supprime les mots anglais très fréquents comme `the`, `and`, `is`, etc. ;
- `max_features=5000` : limite le vocabulaire aux 5000 mots les plus utiles afin d’éviter une matrice trop grande ;
- `sublinear_tf=True` : réduit l’effet des mots répétés trop souvent dans une même critique.

Dans cette version, la vectorisation retenue est TF-IDF. Une comparaison avec `CountVectorizer` pourrait constituer une piste d’amélioration afin d’approfondir l’étude du choix de vectorisation.

---

## 4. Ajout de caractéristiques supplémentaires

En plus de la vectorisation TF-IDF, le script extrait des caractéristiques simples à partir de chaque critique. Ces caractéristiques sont calculées dans la méthode `extraction_features()`.

J’ai également utilisé `spaCy` pour extraire deux informations linguistiques supplémentaires : le nombre d’adjectifs et le nombre d’entités nommées.

Les adjectifs sont pertinents dans des critiques de films, car ils expriment souvent une opinion, par exemple `good`, `bad`, `boring` ou `excellent`.

Les entités nommées peuvent aussi apporter une information sur le contenu de la critique, par exemple lorsqu’un texte mentionne des acteurs, des réalisateurs ou des titres.

Pour chaque texte, les traits suivants sont extraits :

1. nombre de caractères ;
2. nombre de mots ;
3. nombre approximatif de phrases ;
4. nombre de points d’exclamation ;
5. nombre de points d’interrogation ;
6. longueur moyenne des mots ;
7. nombre d’adjectifs ;
8. nombre d’entités nommées.

Exemple de features obtenues pour une critique :

```text
[1896, 356, 19, 0, 0, 4.33, 18, 24]
```

Cela correspond à : 1896 caractères, 356 mots, 19 phrases estimées, 0 point d’exclamation, 0 point d’interrogation, une longueur moyenne de 4.33 caractères par mot, 18 adjectifs et 24 entités nommées.

---

## 5. Préparation des données

La méthode `prepare_data()` combine deux types d’informations :

1. les vecteurs textuels obtenus avec `TfidfVectorizer` ;
2. les features additionnelles extraites manuellement et avec `spaCy`.

La concaténation est réalisée avec :

```python
hstack([X_text, X_features])
```

La matrice finale contient donc :

```text
vecteurs TF-IDF + caractéristiques additionnelles
```

Dans les tests effectués, la forme obtenue est :

```text
Forme de X_text : (602, 5008)
```

Cela correspond à :

- 602 critiques ;
- 5000 colonnes issues du vectoriseur TF-IDF ;
- 8 colonnes correspondant aux features additionnelles.

---

## 6. Modèles testés

Plusieurs modèles de classification sont configurés afin de comparer leurs performances.

Les modèles testés sont :

### SVM

```python
SVC()
```

Hyperparamètres testés :

```python
{
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}
```

Le SVM permet de tester un modèle performant pour la classification. Cependant, l’entraînement peut être assez long avec `SVC`, surtout lorsque le noyau linéaire est utilisé sur des données textuelles de grande dimension.

### Régression logistique

```python
LogisticRegression(max_iter=1000)
```

Hyperparamètres testés :

```python
{
    "C": [0.1, 1, 10],
    "solver": ["liblinear"]
}
```

La régression logistique est adaptée aux problèmes de classification binaire et permet d’obtenir un modèle linéaire interprétable.

### Arbre de décision

```python
DecisionTreeClassifier(random_state=42)
```

Hyperparamètres testés :

```python
{
    "max_depth": [None, 10, 20, 30],
    "criterion": ["gini", "entropy"]
}
```

L’arbre de décision permet de tester une approche non linéaire fondée sur des séparations successives des données.

### Naive Bayes

```python
MultinomialNB()
```

Hyperparamètres testés :

```python
{
    "alpha": [0.1, 0.5, 1.0]
}
```

Le modèle bayésien naïf est souvent utilisé pour la classification de textes, notamment avec des représentations de type comptage ou TF-IDF.

### Forêt aléatoire

```python
RandomForestClassifier(random_state=42)
```

Hyperparamètres testés :

```python
{
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}
```

La forêt aléatoire permet de combiner plusieurs arbres de décision pour obtenir une prédiction plus robuste.

---

## 7. Recherche des meilleurs hyperparamètres

Pour chaque modèle, les hyperparamètres sont testés avec `GridSearchCV`.

La méthode utilisée est :

```python
GridSearchCV(
    estimator=self.model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)
```

`GridSearchCV` permet de tester automatiquement plusieurs combinaisons d’hyperparamètres.

La validation croisée permet d’obtenir une estimation plus fiable des performances qu’un simple découpage train/test.

Le score utilisé est l’accuracy, car le problème est une classification binaire avec deux classes : critiques positives et critiques négatives.

---

## 8. Découpage train/test

Les données sont séparées avec :

```python
train_test_split(
    X_text,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

- `test_size=0.25` : 25 % des données sont utilisées pour le test ;
- `random_state=42` : permet d’obtenir des résultats reproductibles ;
- `stratify=y` : conserve la même proportion de critiques positives et négatives dans les ensembles d’entraînement et de test.

---

## 9. Évaluation des modèles

Après entraînement, chaque modèle est évalué sur l’ensemble de test.

Les métriques affichées sont :

```python
accuracy_score(y_test, y_predict)
classification_report(y_test, y_predict)
```

Le rapport de classification contient :

- la précision ;
- le rappel ;
- le f1-score ;
- le support de chaque classe.

Ces métriques permettent de comparer les modèles de manière plus complète que l’accuracy seule, comme vu dans le TP 8/9.

---

## 10. Instructions d’utilisation

### Installation des dépendances

Créer et activer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
```

Installer les bibliothèques nécessaires :

```bash
pip install scikit-learn matplotlib scipy spacy
python -m spacy download en_core_web_sm
```

### Lancement du programme

Se placer dans le dossier contenant le script et le dossier `imdb_smol` :

```bash
cd chemin/vers/le/dossier/du/projet
```

Lancer le script :

```bash
python TP_final-squelette.py
```

Le programme affiche :

- quelques tests des fonctions réalisées ;
- les résultats obtenus pour chaque modèle.

---

## 11. Résultats obtenus

L’exécution du programme peut prendre un peu de temps, surtout pour le SVM. Cela vient du fait que `GridSearchCV` teste plusieurs combinaisons d’hyperparamètres, et que chaque combinaison est évaluée plusieurs fois grâce à la validation croisée. Le programme ne teste donc pas un seul modèle, mais plusieurs versions d’un même modèle.

Dans cette version, les textes sont représentés avec une vectorisation TF-IDF à laquelle sont ajoutées 8 caractéristiques supplémentaires, dont 2 extraites avec `spaCy` : le nombre d’adjectifs et le nombre d’entités nommées.

Avec une validation croisée à 3 folds, les résultats obtenus sont les suivants :

| Modèle | Meilleurs hyperparamètres | Score validation croisée | Score test |
|---|---:|---:|---:|
| SVM | `C=1`, `kernel='linear'` | 0.749 | 0.834 |
| Régression logistique | `C=10`, `solver='liblinear'` | 0.809 | 0.887 |
| Arbre de décision | `criterion='gini'`, `max_depth=10` | 0.623 | 0.709 |
| Naive Bayes | `alpha=0.1` | 0.758 | 0.828 |
| Forêt aléatoire | `max_depth=None`, `n_estimators=100` | 0.776 | 0.795 |

---

## 12. Analyse des résultats

En regardant les résultats, on remarque que le meilleur modèle est la **régression logistique**. Elle obtient le meilleur score sur les données de test, avec une accuracy d’environ **0.89**. Les meilleurs paramètres trouvés sont `C=10` et `solver='liblinear'`.

La régression logistique fonctionne bien sur les deux classes. Elle obtient un f1-score de **0.89** pour les critiques négatives et de **0.88** pour les critiques positives. Cela montre que le modèle arrive à distinguer correctement les critiques positives et négatives, sans être trop déséquilibré.

Le **SVM** obtient aussi un bon score avec les nouvelles features, avec une accuracy d’environ **0.83**. Son meilleur paramètre est `C=1` avec un noyau `linear`. Le score est meilleur qu’avant l’ajout des features linguistiques, mais l’entraînement reste assez long, surtout pour les combinaisons avec un noyau linéaire. Pour ce type de données, un `LinearSVC` pourrait probablement être plus rapide qu’un `SVC` classique.

Le modèle **Naive Bayes** donne également un résultat correct, avec une accuracy d’environ **0.83**. Son meilleur paramètre est `alpha=0.1`. Il reste intéressant pour la classification de textes, même s’il est ici un peu moins performant que la régression logistique.

La **forêt aléatoire** obtient une accuracy d’environ **0.79**. Son meilleur réglage est `max_depth=None` et `n_estimators=100`. Ce score est correct, mais il reste inférieur aux modèles linéaires. Cela peut s’expliquer par le fait que les données textuelles vectorisées produisent une matrice de grande dimension, ce qui n’est pas toujours idéal pour les modèles basés sur des arbres.

L’**arbre de décision** seul est le modèle le moins performant, avec une accuracy d’environ **0.71**. Même avec une profondeur limitée à 10, il généralise moins bien que les autres modèles. Un seul arbre est souvent plus instable qu’une forêt aléatoire ou qu’un modèle linéaire.

Globalement, l’ajout des features linguistiques avec `spaCy` a modifié les résultats. Le SVM progresse nettement, tandis que la régression logistique reste le modèle le plus performant. Cela montre que les informations ajoutées, comme le nombre d’adjectifs et d’entités nommées, peuvent apporter un complément utile à la représentation TF-IDF.

---

## 13. Choix du meilleur modèle

D’après les résultats obtenus, la meilleure configuration est la vectorisation TF-IDF associée à une régression logistique.

La représentation utilisée est :

```python
TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    sublinear_tf=True
)
```

À cette représentation sont ajoutées 8 features supplémentaires :

- nombre de caractères ;
- nombre de mots ;
- nombre approximatif de phrases ;
- nombre de points d’exclamation ;
- nombre de points d’interrogation ;
- longueur moyenne des mots ;
- nombre d’adjectifs ;
- nombre d’entités nommées.

Le meilleur modèle est :

```python
LogisticRegression(
    C=10,
    solver="liblinear",
    max_iter=1000
)
```

Cette combinaison obtient le meilleur score sur l’ensemble de test, avec une accuracy d’environ **89 %**. Elle est donc retenue comme meilleure configuration dans cette expérience.
