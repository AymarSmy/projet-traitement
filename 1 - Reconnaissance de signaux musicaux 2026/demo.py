import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from algorithm import Encoding, Matching

if __name__ == '__main__':

    # 1: Chargement de la base de données
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    # 2: Configuration de l'encodeur (Paramètres identiques à database.py)
    encoder = Encoding(nperseg=128, noverlap=32, min_distance=25, 
                       time_window=1.0, freq_window=1500)
      
    # 3: Lecture du fichier secret
    filename = 'secret_sample.wav'
    print(f"Analyse de : {filename}")
    
    fs, s = read(filename)
    
    # Gérer la stéréo si nécessaire
    if len(s.shape) > 1:
        s = s[:, 0]
    
    # Normalisation du signal (optionnel mais recommandé)
    s = s.astype(float)

    # 4: Extraction de la signature
    # On traite TOUT le fichier secret, pas juste un petit morceau
    encoder.process(fs, s)
    hashes_extrait = encoder.hashes
    print(f"Nombre de hashs extraits : {len(hashes_extrait)}")

    # 5: Comparaison avec la base de données
    scores = []
    names = []

    print("Comparaison en cours...")
    for entry in database:
        matcher = Matching(hashes_extrait, entry['hashcodes'])
        score = matcher.get_score()
        scores.append(score)
        names.append(entry['song'])

    # 6: Résultat final
    best_idx = np.argmax(scores)
    print("\n" + "="*30)
    print(f"RÉSULTAT : {names[best_idx]}")
    print(f"SCORE : {scores[best_idx]}")
    print("="*30)

    # Affichage de l'histogramme pour le gagnant
    best_matcher = Matching(hashes_extrait, database[best_idx]['hashcodes'])
    best_matcher.display_histogram()