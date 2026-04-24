import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import os

# On utilise os.path.join pour que Windows ne se trompe pas de chemin
# On prend le fichier secret_sample.wav qui est juste à côté de ton script
sample_file = 'secret_sample.wav' 

# Lecture du fichier
fs, s = wavfile.read(sample_file)

# Conversion en mono si nécessaire
if len(s.shape) > 1:
    s = s[:, 0]

# Calcul du spectrogramme avec les paramètres de ton projet
f, t, S = spectrogram(s, fs, nperseg=128, noverlap=32)

# Affichage
plt.figure(figsize=(10, 5))
# Utilisation du log pour les décibels
plt.pcolormesh(t, f, 10 * np.log10(S + 1e-10), shading='gouraud', cmap='viridis')

plt.title(f'Spectrogramme : {sample_file}')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
plt.colorbar(label='Intensité [dB]')
plt.ylim(0, 5000) # On se limite à 5000Hz pour la lisibilité musicale
plt.tight_layout()
plt.show()

# Transformation en dB
S_db = 10 * np.log10(S + 1e-10)

# --- MODIFICATION ICI POUR LE CONTRASTE ---
# On définit un seuil : on ne garde que ce qui est au-dessus de -20dB par exemple
# vmax = le point le plus brillant, vmin = le seuil de noirceur
v_max = np.max(S_db)
v_min = v_max - 40  # On affiche seulement une plage de 40dB sous le maximum
# ------------------------------------------

plt.figure(figsize=(10, 5))
# On ajoute vmin et vmax pour le contraste
plt.pcolormesh(t, f, S_db, shading='gouraud', cmap='viridis', vmin=v_min, vmax=v_max)

plt.title(f'Spectrogramme Contrasté : {sample_file}')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
plt.colorbar(label='Intensité [dB]')
plt.ylim(0, 5000) 
plt.tight_layout()
plt.show()