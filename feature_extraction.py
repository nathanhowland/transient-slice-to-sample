import os
import numpy as np
from scipy.io import wavfile

# Function to load audio file using scipy
def load_audio(file_path):
    rate, audio_data = wavfile.read(file_path)
    return audio_data, rate

# Function to calculate spectral rolloff
def spectral_rolloff(audio_data, rate, rolloff_percent=0.85):
    stft = np.abs(np.fft.fft(audio_data))
    total_energy = np.sum(stft)
    target_energy = rolloff_percent * total_energy
    cumulative_sum = np.cumsum(stft)
    rolloff_freq = np.argmax(cumulative_sum > target_energy)
    return rolloff_freq * rate / len(audio_data)

# Function to calculate spectral contrast
def spectral_contrast(audio_data, rate, n_bands=6):
    stft = np.abs(np.fft.fft(audio_data))
    if stft.ndim == 1:
        stft = stft.reshape(-1, 1)  # Ensure stft has 2D shape for processing
    freq_bins, time_bins = stft.shape
    freq_midpoints = np.linspace(0, rate / 2, num=freq_bins, endpoint=True)
    band_widths = np.linspace(0, rate / 2, num=n_bands + 1, endpoint=True)
    contrasts = np.zeros((n_bands, time_bins))
    for i in range(n_bands):
        band_start, band_end = int(band_widths[i]), int(band_widths[i + 1])
        band_stft = stft[band_start:band_end, :]
        
        if band_stft.size == 0 or np.isnan(band_stft).all():
            # Handle empty or NaN arrays
            contrasts[i, :] = 0.0  # Set contrast to 0 if no valid data
        else:
            contrast_band = np.max(band_stft, axis=0) - np.min(band_stft, axis=0)
            contrasts[i, :] = contrast_band
    return contrasts

# Function to calculate spectral bandwidth
def spectral_bandwidth(audio_data, rate):
    stft = np.abs(np.fft.fft(audio_data))
    freqs = np.fft.fftfreq(len(audio_data), 1 / rate)
    spectral_centroid = np.sum(freqs * stft) / np.sum(stft)
    bandwidth = np.sum(stft * np.square(freqs - spectral_centroid)) / np.sum(stft)
    return bandwidth

# Function to calculate spectral centroid
def spectral_centroid(audio_data, rate):
    stft = np.abs(np.fft.fft(audio_data))
    freqs = np.fft.fftfreq(len(audio_data), 1 / rate)
    spectral_centroid = np.sum(freqs * stft) / np.sum(stft)
    return spectral_centroid

# Directory containing audio files
audio_dir = '/Users/nathan/Documents/pyproj/sample_slice/output'

# Initialize a dictionary to store results
results = {}

# Iterate over each audio file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(audio_dir, filename)
        
        # Load audio file
        audio_data, rate = load_audio(file_path)
        
        # Calculate features
        rolloff = spectral_rolloff(audio_data, rate)
        contrast = spectral_contrast(audio_data, rate)
        bandwidth = spectral_bandwidth(audio_data, rate)
        centroid = spectral_centroid(audio_data, rate)
        
        # Store results in dictionary
        results[filename] = {
            'spectral_rolloff': rolloff,
            'spectral_contrast': contrast.tolist(),  # Convert numpy array to list
            'spectral_bandwidth': bandwidth,
            'spectral_centroid': centroid
        }

# Output text file
output_file = 'audio_features.txt'

# Save dictionary to text file
with open(output_file, 'w') as f:
    for filename, features in results.items():
        f.write(f'{filename}\n')
        for feature, value in features.items():
            if isinstance(value, list):  # Convert list to string if necessary
                value_str = ', '.join(map(str, value))
            else:
                value_str = str(value)
            f.write(f'- {feature}: {value_str}\n')
        f.write('\n')

print(f'Feature extraction completed. Results saved to {output_file}')
