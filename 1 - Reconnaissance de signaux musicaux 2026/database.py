"""
Create a database containing the hashcodes of the songs stored 
in the specified folder (.wav files only). 
The database is saved as a pickle file as a list of dictionaries.
Each dictionary has two keys 'song' and 'hashcodes', corresponding 
to the name of the song and to the hashcodes used as signature for 
the matching algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
import pickle
from algorithm import Matching
from algorithm import Encoding


# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    folder = './samples/'

    # 1: Load the audio files
    import os
    audiofiles = os.listdir(folder)
    audiofiles = [item for item in audiofiles if item[-4:] =='.wav']

    # 2: Set the parameters of the encoder
    nperseg=128
    noverlap=32
    min_distance=50
    time_window=1.
    freq_window=1500
    encoder = Encoding(nperseg=nperseg, noverlap=noverlap, 
      min_distance=min_distance,
      time_window=time_window, 
      freq_window=freq_window)

    # 3: Construct the database
    database = []
    for audiofile in audiofiles:
        try:
            fs, s = read(os.path.join(folder, audiofile))
            
            # --- CORRECTION 1: Gérer la stéréo ---
            # Si le fichier a 2 colonnes (stéréo), on n'en garde qu'une (mono)
            if len(s.shape) > 1:
                s = s[:, 0]
            
            # --- CORRECTION 2: Vérifier la longueur ---
            # Si le fichier est trop court (moins que nperseg), on l'ignore
            if len(s) <= nperseg:
                print(f"Saut de {audiofile} : fichier trop court ou vide.")
                continue

            print(f'Song: {audiofile[:-4]}')
            
            # Encodage
            encoder.process(fs, s)
            
            database.append({
                'song': audiofile,
                'hashcodes': encoder.hashes
            })
            
        except Exception as e:
            print(f"Erreur sur le fichier {audiofile}: {e}")
            continue

    # 4: Save the database
    with open('songs.pickle', 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

