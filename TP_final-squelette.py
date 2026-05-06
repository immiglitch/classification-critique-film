import os
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# compléter avec toutes les librairies sklearn utiles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

#import pour les modeles
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack



class entrainement_modele:
    def __init__(self, model, directory='imdb_smol'):
        self.model = model
        self.directory = directory
        
        # Pipeline spaCy pour extraire des informations linguistiques
        self.nlp = spacy.load("en_core_web_sm")
        
        # Vectoriseur choisi pour transformer les textes en données numériques
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        
    #Charge et extrait les textes et les labels des fichiers.
        
    def get_data(self):
        # renvoie deux listes : la liste des textes et la liste des labels correspondants 
        textes, y = [], []
        
        for label in ["neg", "pos"]:
            dossier = os.path.join(self.directory, label)
            
            for fichier in os.listdir(dossier):
                chemin_fichier = os.path.join(dossier, fichier)
                
                if os.path.isfile(chemin_fichier):
                    with open(chemin_fichier, "r", encoding="utf-8", errors="ignore") as f:
                        textes.append(f.read())
                        y.append(label)
        
        return textes, y
    
    #Extrait des caracteristiques linguistiques a partir des textes.
    
    def extraction_features(self, textes):
        features = []
        # à compléter en cas d'ajout de features additionnelles
        
        for doc in self.nlp.pipe(textes, disable=["parser"], batch_size=20):
            texte = doc.text
            mots = texte.split()
            
            nombre_caracteres = len(texte)
            nombre_mots = len(mots)
            nombre_phrases = texte.count(".") + texte.count("!") + texte.count("?")
            nombre_exclamations = texte.count("!")
            nombre_questions = texte.count("?")
            
            if nombre_mots > 0:
                longueur_moyenne_mots = sum(len(mot) for mot in mots) / nombre_mots
            else:
                longueur_moyenne_mots = 0
            
            # Feature spaCy : nombre d'adjectifs
            nombre_adjectifs = sum(1 for token in doc if token.pos_ == "ADJ")
            
            # Feature spaCy : nombre d'entités nommées
            nombre_entites_nommees = len(doc.ents)
            
            features.append([
                nombre_caracteres,
                nombre_mots,
                nombre_phrases,
                nombre_exclamations,
                nombre_questions,
                longueur_moyenne_mots,
                nombre_adjectifs,
                nombre_entites_nommees
            ])
            
        return features
        

    #Prepare les donnees pour la formation en combinant les vecteurs 
    def prepare_data(self):
        textes, y = self.get_data()
        
        # construction des vecteurs X_text en transformant les textes en nombres
        X_text = self.vectorizer.fit_transform(textes)
        
        # Extraction des features additionnelles
        X_features = self.extraction_features(textes)
        
        # Concaténation du vectorizer avec les autres features
        X_text = hstack([X_text, X_features])
        
        return X_text, y

    #Entraîne le modele en utilisant une recherche sur grid pour trouver les meilleurs parametres.

    def grid_search_train(self, param_grid):
        X_text, y = self.prepare_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_text,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        y_predict = grid_search.predict(X_test)
        
        print("Meilleurs paramètres :", grid_search.best_params_)
        print("Meilleur score en validation croisée :", grid_search.best_score_)
        print("Score sur les données de test :", accuracy_score(y_test, y_predict))
        print(classification_report(y_test, y_predict))

# Configuration des modeles avec leurs hyperparametres pour la recherche sur grid.
# à compléter avec les autres modèles
modeles = {
    'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Regression logistique': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10], 'solver': ['liblinear']}),
    'Arbre de decision': (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20, 30], 'criterion': ['gini', 'entropy']}),
    'Naive Bayes': (MultinomialNB(), {'alpha': [0.1, 0.5, 1.0]}),
    'Foret aleatoire': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]})
}

if __name__ == "__main__":
    test = entrainement_modele(model=SVC())
    # test des fonctions de la classe train model

    print("Chemin du dossier utilisé :")
    print(test.directory)

    # test get data
    textes, y = test.get_data()

    print("Nombre de textes chargés :", len(textes))
    print("Nombre de labels chargés :", len(y))
    print("Labels trouvés :", set(y))

    print("\nExtrait du premier texte :")
    print(textes[0][:500])

    # test extraction features
    features = test.extraction_features(textes)

    print("\nNombre de lignes de features :", len(features))
    print("Nombre de features pour le premier texte :", len(features[0]))

    print("\nFeatures du premier texte :")
    print(features[0])

    # test prepare data 
    X_text, y = test.prepare_data()

    print("\nForme de X_text :", X_text.shape)
    print("Nombre de labels :", len(y))
    print("Labels trouvés :", set(y))

    # entraînement des modèles
    for model_name, (model, params) in modeles.items():
        classifier = entrainement_modele(model=model)
        print(f"\nentrainement de {model_name}")
        classifier.grid_search_train(params)