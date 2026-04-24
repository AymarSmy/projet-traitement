import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from algorithm import Encoding, Matching

# ----------------------------------------------
# Run the script
# ----------------------------------------------

if __name__ == '__main__':

    # 1: Chargement de la base de données
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    # 2: Encoder
    nperseg=128
    noverlap=32
    min_distance=25
    time_window=1.
    freq_window=1500
    encoder = Encoding(nperseg=nperseg, noverlap=noverlap, 
      min_distance=min_distance,
      time_window=time_window, 
      freq_window=freq_window)
    
    # 3: Randomly get an extract from one of the songs of the database
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tstart = np.random.randint(20, 90)
    tmin = int(tstart*fs)
    duration = int(10*fs)

    # 4: Use the encoder to extract a signature from the extract
    encoder.process(fs, s[tmin:tmin + duration])
    hashes = encoder.hashes

    # 5: Comparaison avec la base de données
    scores = []
    names = []


    for entry in database:
        matcher = Matching(hashes, entry['hashcodes'])
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
    best_matcher = Matching(hashes, database[best_idx]['hashcodes'])
    best_matcher.display_histogram()

    #Affichage histogramme pour une autre chanson
    no_matcher = Matching(hashes, database[(best_idx + 1) % len(database)]['hashcodes'])
    no_matcher.display_histogram()