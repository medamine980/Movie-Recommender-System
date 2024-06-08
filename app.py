from flask import Flask, Response, request
from flask import render_template
from utils import trouver_similaires_films, nettoyer_donnees, rechercher
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import requests
import re

API_KEYS = ['c7ec19ffdd3279641fb606d19ceb9bb1', '3e60d738f7660ca7c31aaa64f2f25bc0']

def fetch_poster(film_nom):
     url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(film_nom)
     donnee=requests.get(url)
     donnee=donnee.json()
     poster_chemin = donnee['poster_path']
     full_chemin = "https://image.tmdb.org/t/p/w500/"+poster_chemin
     return full_chemin


app = Flask(__name__)

films = pd.read_csv("./data/ml-25m/movies.csv")
notes = pd.read_csv("./data/ml-25m/ratings.csv")
liens = pd.read_csv('./data/ml-25m/links.csv')

films['tmdbId'] = liens['tmdbId']

films["titre_propre"] = films["title"].apply(nettoyer_donnees)

# ngram c'est-à-dire le nombre de mots à prendre pour faire 
# la comparaison (la recherche)
# Exemple: ngram=(1,2) prendre 1 ou 2 mots pour faire la recherche
vectoriseur = TfidfVectorizer(ngram_range=(1,2))

# La conversion du colonne "titre_propre" en un tableau de nombres
tfidf = vectoriseur.fit_transform(films["titre_propre"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/autocomplete")
def autocomplete():
    recherche = request.args.get('recherche')
    films_rechercher = rechercher(films, tfidf, vectoriseur, recherche)
    films_rechercher['image'] = films_rechercher['tmdbId'].apply(lambda tmdbId: fetch_poster(tmdbId))
    return Response(films_rechercher.to_json(orient="records"), mimetype='application/json')

@app.route("/api/recommendations")
def recommendations():
    film_id = request.args.get('film_id')
    films_recommendes = trouver_similaires_films(films, notes, int(film_id))
    films_recommendes['image'] = films_recommendes['tmdbId'].apply(lambda tmdbId: fetch_poster(tmdbId))
    return Response(films_recommendes.to_json(orient="records"), mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0')