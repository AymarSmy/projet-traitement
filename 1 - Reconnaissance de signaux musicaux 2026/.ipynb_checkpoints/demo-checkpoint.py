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

# +
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
    
    # On choisit un morceau
    filename = './samples/' + '592.wav'

    fs, s = read(filename)
    #toujours pour régler le problème de stéréo
    if len(s.shape) > 1:
        s = s[:, 0]
    #sinon on a des problèmes d'extrait trop courts
    total_duration = len(s) / fs
    segment_length = 10
    if total_duration > segment_length:
        tstart = np.random.uniform(0, total_duration - segment_length)
    else:
        tstart = 0
        segment_length = total_duration
    tmin = int(tstart*fs)
    duration = int(10*fs)

    encoder.process(fs, s[tmin:tmin + duration])
    hashes1 = encoder.hashes
    
    # On choisit le même morceau
    filename2 = './samples/' + '592.wav'

    fs2, s2 = read(filename2)
    #toujours pour régler le problème de stéréo
    if len(s2.shape) > 1:
        s2 = s2[:, 0]
    #sinon on a des problèmes d'extrait trop courts
    total_duration2 = len(s2) / fs2
    segment_length2 = 10
    if total_duration2 > segment_length2:
        tstart2 = np.random.uniform(0, total_duration2 - segment_length2)
    else:
        tstart2 = 0
        segment_length2 = total_duration2
    tmin2 = int(tstart2*fs2)
    duration2 = int(10*fs2)

    encoder.process(fs2, s2[tmin2:tmin2 + duration2])
    hashes2 = encoder.hashes



    matcher = Matching(hashes1, hashes2)
    score = matcher.get_score()

    # Résultat final

    print("\n" + "="*30)
    print("Bon matching")
    print(f"SCORE : {score}")
    print("="*30)

    #Affichage du nuage de points pour le gagnant (question 6)
    matcher.display_scatterplot()

    # Affichage de l'histogramme pour le gagnant (question 6)
    matcher.display_histogram()




# +
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
    
    # On choisit un morceau
    filename = './samples/' + '592.wav'

    fs, s = read(filename)
    #toujours pour régler le problème de stéréo
    if len(s.shape) > 1:
        s = s[:, 0]
    #sinon on a des problèmes d'extrait trop courts
    total_duration = len(s) / fs
    segment_length = 10
    if total_duration > segment_length:
        tstart = np.random.uniform(0, total_duration - segment_length)
    else:
        tstart = 0
        segment_length = total_duration
    tmin = int(tstart*fs)
    duration = int(10*fs)

    encoder.process(fs, s[tmin:tmin + duration])
    hashes1 = encoder.hashes
    
    # On choisit le même morceau
    filename2 = './samples/' + '368.wav'

    fs2, s2 = read(filename2)
    #toujours pour régler le problème de stéréo
    if len(s2.shape) > 1:
        s2 = s2[:, 0]
    #sinon on a des problèmes d'extrait trop courts
    total_duration2 = len(s2) / fs2
    segment_length2 = 10
    if total_duration2 > segment_length2:
        tstart2 = np.random.uniform(0, total_duration2 - segment_length2)
    else:
        tstart2 = 0
        segment_length2 = total_duration2
    tmin2 = int(tstart2*fs2)
    duration2 = int(10*fs2)

    encoder.process(fs2, s2[tmin2:tmin2 + duration2])
    hashes2 = encoder.hashes



    matcher = Matching(hashes1, hashes2)
    score = matcher.get_score()

    # Résultat final

    print("\n" + "="*30)
    print("Mauvais matching")
    print(f"SCORE : {score}")
    print("="*30)

    #Affichage du nuage de points pour le gagnant (question 6)
    matcher.display_scatterplot()

    # Affichage de l'histogramme pour le gagnant (question 6)
    matcher.display_histogram()



# +
##Vérification de la robustesse de l'algorithme

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
    #toujours pour régler le problème de stéréo
    if len(s.shape) > 1:
        s = s[:, 0]
    #sinon on a des problèmes d'extrait trop courts
    total_duration = len(s) / fs
    segment_length = 10
    if total_duration > segment_length:
        tstart = np.random.uniform(0, total_duration - segment_length)
    else:
        tstart = 0
        segment_length = total_duration
    tmin = int(tstart*fs)
    duration = int(10*fs)

    encoder.process(fs, s[tmin:tmin + duration])
    hashes = encoder.hashes

    # Comparaison avec la base de données
    scores = []
    names = []


    for entry in database:
        matcher = Matching(hashes, entry['hashcodes'])
        score = matcher.get_score()
        scores.append(score)
        names.append(entry['song'])

    # Résultat final
    best_idx = np.argmax(scores)
    print("\n" + "="*30)
    print(f"RÉSULTAT : {names[best_idx]}")
    print(f"SCORE : {scores[best_idx]}")
    print("="*30)


# +
## Recherche du morceau secret

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
    # On ouvre le morceau secret
    filename = './secret_sample.wav'

    fs, s = read(filename)
    #toujours pour régler le problème de stéréo
    if len(s.shape) > 1:
        s = s[:, 0]
    #sinon on a des problèmes d'extrait trop courts
    total_duration = len(s) / fs
    segment_length = 10
    if total_duration > segment_length:
        tstart = np.random.uniform(0, total_duration - segment_length)
    else:
        tstart = 0
        segment_length = total_duration
    tmin = int(tstart*fs)
    duration = int(10*fs)


    encoder.process(fs, s[tmin:tmin + duration])
    hashes = encoder.hashes

    #Comparaison avec la base de données
    scores = []
    names = []


    for entry in database:
        matcher = Matching(hashes, entry['hashcodes'])
        score = matcher.get_score()
        scores.append(score)
        names.append(entry['song'])

    # Résultat final
    best_idx = np.argmax(scores)
    print("\n" + "="*30)
    print(f"RÉSULTAT : {names[best_idx]}")
    print(f"SCORE : {scores[best_idx]}")
    print("="*30)
# -


