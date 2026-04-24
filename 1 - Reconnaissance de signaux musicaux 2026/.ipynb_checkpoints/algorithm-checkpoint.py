"""
Algorithm implementation
"""
"""
Algorithm implementation - Complété
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

class Encoding:

    """
    Class implementing the procedure for creating a fingerprint 
    for the audio files

    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """

    def __init__(self, nperseg=128, noverlap=32, min_distance=50, time_window=1.0, freq_window=1500):
        # Paramètres du spectrogramme
        self.nperseg = nperseg
        self.noverlap = noverlap
        # Paramètres de la constellation
        self.min_distance = min_distance
        # Paramètres du hachage
        self.delta_t = time_window
        self.delta_f = freq_window
        
        # Attributs qui seront calculés
        self.f = None
        self.t = None
        self.S = None
        self.anchors = None
        self.hashes = []

    def process(self, fs, s):
        """
        Calcule le spectrogramme, extrait la constellation et crée les hashs.
        """
        self.fs = fs
        self.s = s

        # 1. Calcul du spectrogramme
        self.f, self.t, self.S = spectrogram(self.s, self.fs, 
                                             nperseg=self.nperseg, 
                                             noverlap=self.noverlap)

        # 2. Extraction de la constellation (maxima locaux)
        # peak_local_max travaille sur des indices de matrice
        coordinates = peak_local_max(self.S, 
                                     min_distance=self.min_distance, 
                                     exclude_border=False)
        
        # Conversion des indices en unités physiques (temps, fréquence)
        # coordinates[:, 0] -> axe des fréquences (f), coordinates[:, 1] -> axe du temps (t)
        self.anchors = np.zeros(coordinates.shape)
        self.anchors[:, 0] = self.t[coordinates[:, 1]] # Temps
        self.anchors[:, 1] = self.f[coordinates[:, 0]] # Fréquence

        # 3. Création des codes de hachage
        self.hashes = []
        num_peaks = len(self.anchors)
        
        for i in range(num_peaks):
            t_a, f_a = self.anchors[i]
            for j in range(num_peaks):
                t_i, f_i = self.anchors[j]
                
                # Critères cible : 0 < ti - ta <= Delta_t ET |fa - fi| < Delta_f
                diff_t = t_i - t_a
                diff_f = abs(f_a - f_i)
                
                if 0 < diff_t <= self.delta_t and diff_f < self.delta_f:
                    self.hashes.append({
                        "t": t_a,
                        "hash": np.array([diff_t, f_a, f_i])
                    })

    def display_spectrogram(self, display_anchors=True):
        plt.figure(figsize=(10, 6))
        # Utilisation de log10 pour mieux voir les pics d'énergie
        plt.pcolormesh(self.t, self.f/1e3, 10 * np.log10(self.S + 1e-10), shading='gouraud')
        plt.xlabel('Temps [s]')
        plt.ylabel('Fréquence [kHz]')
        plt.title('Spectrogramme et Constellation')
        if display_anchors and self.anchors is not None:
            plt.scatter(self.anchors[:, 0], self.anchors[:, 1]/1e3, 
                        color='red', s=5, label='Maxima (Constellation)')
            plt.legend()
        plt.colorbar(label='Puissance (dB)')
        plt.show()


class Matching:
    """
    Compare the hashes from two audio files to determine if these
    files match

    Attributes
    ----------

    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries
       hashes extracted as fingerprint for the second audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """
    def __init__(self, hashes1, hashes2):
        """
        Compare the hashes from two audio files to determine if these
        files match

        Parameters
        ----------

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
           hashes extracted as fingerprint for the second audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target
          
        """
        self.hashes1 = hashes1 # Extrait
        self.hashes2 = hashes2 # Base de données

        # Préparation pour la comparaison rapide
        times1 = np.array([item['t'] for item in self.hashes1])
        hashcodes1 = np.array([item['hash'] for item in self.hashes1])
        
        times2 = np.array([item['t'] for item in self.hashes2])
        hashcodes2 = np.array([item['hash'] for item in self.hashes2])

        self.matching = []
        # Pour chaque hash de l'extrait, on cherche s'il existe dans le morceau
        for i, h1 in enumerate(hashcodes1):
            # On cherche les indices où les vecteurs (dt, fa, fi) sont identiques
            dist = np.sum(np.abs(hashcodes2 - h1), axis=1)
            mask = (dist < 1e-6)
            if mask.any():
                # On stocke le couple (ta_extrait, ta_morceau)
                for t2 in times2[mask]:
                    self.matching.append(np.array([times1[i], t2]))
        
        self.matching = np.array(self.matching)
        
        # 1. Calcul des offsets (ta_morceau - ta_extrait)
        if len(self.matching) > 0:
            self.offsets = self.matching[:, 1] - self.matching[:, 0]
        else:
            self.offsets = np.array([])

    def get_score(self):
        """
        Critère de décision : le pic maximal de l'histogramme des offsets.
        """
        if len(self.offsets) == 0:
            return 0
        counts, _ = np.histogram(self.offsets, bins=100)
        return np.max(counts)

    def display_scatterplot(self):
        """
        Display through a scatterplot the times associated to the hashes
        that match
        """
        if len(self.matching) > 0:
            plt.scatter(self.matching[:, 0], self.matching[:, 1], s=10, alpha=0.5)
            plt.xlabel('Temps dans l\'extrait (s)')
            plt.ylabel('Temps dans le morceau (s)')
            plt.title('Nuage de points des correspondances')
            plt.show()

    def display_histogram(self):
        if len(self.offsets) > 0:
            plt.hist(self.offsets, bins=100)
            plt.xlabel('Offset (s)')
            plt.ylabel('Nombre de correspondances')
            plt.title('Histogramme des décalages temporels')
            plt.show()