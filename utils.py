from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re


# Nettoyage
def nettoyer_donnees(titre):
    return re.sub('[^a-zA-Z0-9 ]', '', titre)


def rechercher(films, tfidf, vectoriseur, titre):
    # nettoyage du titre
    titre = nettoyer_donnees(titre)
    # le convertir en un tableau de nombres
    vecteur = vectoriseur.transform([titre])
    # trouver les similitudes
    similitudes = cosine_similarity(vecteur, tfidf).flatten()
    indices = np.argpartition(similitudes, -5)[-5:]
    resultats = films.iloc[indices][::-1]
    return resultats

def trouver_similaires_films(films, notes, film_id):
    # Premierement on a besoin de trouver les utilisateurs qui aiment le film qu'on
    # cherche
    utilisateurs_similaires = notes[(notes["movieId"] == film_id) & (
        notes["rating"] > 4)]["userId"]
    
    # aprés trouver les autres films que ces utilisateurs aiment aussi
    films_aime = notes[(notes["userId"].isin(utilisateurs_similaires)) & 
                   (notes["rating"] > 4)]["movieId"]
    
    # Trouver le pourcentage de chaque film aimé par rapport aux nombres d'utilisateurs
    # similaires
    films_aime = films_aime.value_counts() / len(utilisateurs_similaires)
    
    # Filtrer alors pour seuls les pourcentages qui s'afficheront sont supérieurs à 10%
    films_aime_pourcentage = films_aime[films_aime > 0.1]
    
    # Trouver touts les utilisateurs qui ont aimé les "films_aimé"
    toutes_utilisateurs = notes[notes["movieId"].isin(films_aime_pourcentage.index) & 
                                (notes["rating"] > 4)]
    
    # Trouver le pourcentage de touts les utilisateurs qui ont aimé chaque film
    toutes_utilisateurs_percentage = toutes_utilisateurs["movieId"].value_counts()/len(
        toutes_utilisateurs["userId"]
    )
    
    tab_pourcentages = pd.concat([films_aime_pourcentage, toutes_utilisateurs_percentage], axis=1)
    tab_pourcentages.columns = ["Similaires", "Touts"]
    
    tab_pourcentages["score"] = (
        tab_pourcentages["Similaires"] / tab_pourcentages["Touts"]
    )
    
    tab_pourcentages = tab_pourcentages.sort_values("score", ascending=False)
    
    return tab_pourcentages.head(10).merge(films, left_index=True, 
                                           right_on="movieId")[
        ["score", "title", "genres", "tmdbId"]
    ]