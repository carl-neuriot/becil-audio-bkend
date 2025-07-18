from datetime import timedelta
from scipy.io import wavfile
from scipy.signal import correlate
from scipy import signal
import librosa.display
from sklearn.preprocessing import normalize
import librosa
import numpy as np
import os


def seconds_to_standard_time(seconds):
    return str(timedelta(seconds=seconds)).split(".")[0]


def extract_brand_name(filename):
    if "_" in filename:
        return filename.split("_")[0]
    else:
        return os.path.splitext(filename)[0]


def load_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path} with librosa: {e}")
        try:
            sr, audio = wavfile.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if audio.dtype != np.float32 and audio.dtype != np.float64:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
            return audio, sr
        except Exception as e2:
            print(f"Error loading {file_path} with scipy: {e2}")
            return None, None


def preprocess_audio(audio, sr):
    """Preprocess audio for better matching"""
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist

    if low < 1.0 and high < 1.0:
        b, a = signal.butter(4, [low, high], btype="band")
        audio = signal.filtfilt(b, a, audio)

    return audio


def extract_mfcc_features(audio, sr, n_mfcc=13):
    """Extract MFCC features for better audio matching"""
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    return features


def compute_feature_correlation(master_features, recording_features):
    """Compute correlation between feature vectors"""
    master_norm = normalize(master_features, axis=0)
    recording_norm = normalize(recording_features, axis=0)

    correlations = []
    for i in range(master_norm.shape[0]):
        corr = signal.correlate(recording_norm[i], master_norm[i], mode="full")
        correlations.append(corr)

    avg_correlation = np.mean(correlations, axis=0)
    return avg_correlation


def normalize_signal(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)


def find_matches_improved(
    master_audio, master_sr, radio_audio, radio_sr, threshold=0.65
):
    """
    Improved ad detection using normalized cross-correlation.
    Works with .mp3 or .wav master files.
    """

    if master_sr != radio_sr:
        master_audio = librosa.resample(
            master_audio, orig_sr=master_sr, target_sr=radio_sr
        )
        master_sr = radio_sr

    master_audio = normalize_signal(master_audio)
    radio_audio = normalize_signal(radio_audio)

    correlation = correlate(radio_audio, master_audio, mode="valid")
    correlation /= len(master_audio)

    matches = []
    ad_duration = len(master_audio) / radio_sr

    i = 0
    while i < len(correlation):
        if correlation[i] >= threshold:
            start_time = i / radio_sr
            end_time = start_time + ad_duration
            matches.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": ad_duration,
                    "correlation": float(round(correlation[i], 4)),
                }
            )

            i += int(ad_duration * radio_sr)
        else:
            i += 1

    return matches
