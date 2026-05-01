import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

def load_wav_mono(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return sr, x

def extract_features(path):
    sr, x = load_wav_mono(path)

    # Basic time-domain features
    rms = np.sqrt(np.mean(x ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2

    # Frequency-domain features using spectrogram
    f, t, Sxx = spectrogram(x, fs=sr, nperseg=512, noverlap=256)
    Sxx = np.log1p(Sxx)

    # Energy bands relevant to chewing/click-like sounds
    def band_energy(low, high):
        mask = (f >= low) & (f < high)
        return float(np.mean(Sxx[mask])) if np.any(mask) else 0.0

    bands = [
        band_energy(0, 300),
        band_energy(300, 800),
        band_energy(800, 1500),
        band_energy(1500, 2500),
        band_energy(2500, 4000),
        band_energy(4000, 7000),
    ]

    # Spectral centroid approximation
    spectrum = np.mean(Sxx, axis=1)
    centroid = float(np.sum(f * spectrum) / (np.sum(spectrum) + 1e-9))

    # Pulse proxy: number of high-energy frames
    frame_energy = np.mean(Sxx, axis=0)
    pulse_proxy = float(np.mean(frame_energy > (np.mean(frame_energy) + 1.5 * np.std(frame_energy))))

    return np.array([rms, zcr, centroid, pulse_proxy] + bands, dtype=np.float32)
