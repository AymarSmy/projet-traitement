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

        To Do
        -----

        This function takes as input a sampled signal s and the sampling
        frequency fs and returns the fingerprint (the hashcodes) of the signal.
        The fingerprint is created through the following steps
        - spectrogram computation
        - local maxima extraction
        - hashes creation

        Implement all these operations in this function. Keep as attributes of
        the class the spectrogram, the range of frequencies, the anchors, the 
        list of hashes, etc.

        Each hash can conveniently be represented by a Python dictionary 
        containing the time associated to its anchor (key: "t") and a numpy 
        array with the difference in time between the anchor and the target, 
        the frequency of the anchor and the frequency of the target 
        (key: "hash")


        Parameters
        ----------

        fs: int
           sampling frequency [Hz]
        s: numpy array
           sampled signal
        """
        self.fs = fs
        self.s = s

        # Calcul du spectrogramme
        self.f, self.t, self.S = spectrogram(self.s, self.fs, 
                                             nperseg=self.nperseg, 
                                             noverlap=self.noverlap)

        #Calcul de la constellation (maxima locaux)
        coordinates = peak_local_max(self.S, 
                                     min_distance=self.min_distance, 
                                     exclude_border=False)
        
        # Conversion des indices en unités physiques (temps, fréquence)
        self.anchors = np.zeros(coordinates.shape)
        self.anchors[:, 0] = self.t[coordinates[:, 1]] # Temps
        self.anchors[:, 1] = self.f[coordinates[:, 0]] # Fréquence

        #Création des codes de hachage
        self.hashes = []
        num_peaks = len(self.anchors)
        
        for i in range(num_peaks):
            t_a, f_a = self.anchors[i]
            for j in range(num_peaks):
                t_i, f_i = self.anchors[j]
                
                diff_t = t_i - t_a
                diff_f = abs(f_a - f_i)
                
                # on respecte bien les critères donnés dans l'énoncé
                if 0 < diff_t <= self.delta_t and diff_f < self.delta_f:
                    self.hashes.append({
                        "t": t_a,
                        "hash": np.array([diff_t, f_a, f_i])
                    })

    def display_spectrogram(self, display_anchors=True):
        """
        Display the spectrogram of the audio signal

        Parameters
        ----------
        display_anchors: boolean
           when set equal to True, the anchors are displayed on the
           spectrogram
        """
        plt.figure(figsize=(10, 6))
        # On utilise le log10 pour mieux voir les pics d'énergie
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
        self.hashes1 = hashes1 # Extrait à identifier
        self.hashes2 = hashes2 # Base de données

        times1 = np.array([item['t'] for item in self.hashes1])
        hashcodes1 = np.array([item['hash'] for item in self.hashes1])
        
        times2 = np.array([item['t'] for item in self.hashes2])
        hashcodes2 = np.array([item['hash'] for item in self.hashes2])

        self.matching = []
        # Pour chaque hash de l'extrait, on cherche s'il existe dans le morceau
        for i, h1 in enumerate(hashcodes1):
            dist = np.sum(np.abs(hashcodes2 - h1), axis=1)
            mask = (dist < 1e-6)
            if mask.any():
                for t2 in times2[mask]:
                    self.matching.append(np.array([times1[i], t2]))
        
        self.matching = np.array(self.matching)
        
        # On calcule les offsets (ta_morceau - ta_extrait)
        if len(self.matching) > 0:
            self.offsets = self.matching[:, 1] - self.matching[:, 0]
        else:
            self.offsets = np.array([])

    def get_score(self):
        """
        On choisit pour critère de décision : le pic maximal de l'histogramme des offsets.
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


