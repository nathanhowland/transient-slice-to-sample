from pydub import AudioSegment
from scipy.signal import find_peaks
import numpy as np
import os

def split_audio_directory(input_dir, output_dir, threshold=-30.0, min_silence_len=100):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize global slice counter
    global_slice_count = 0
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):  # Process only WAV files
            file_path = os.path.join(input_dir, filename)
            global_slice_count = split_audio(file_path, output_dir, threshold, min_silence_len, global_slice_count)

def split_audio(file_path, output_dir, threshold=-30.0, min_silence_len=100, start_slice_index=0):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)

    # Convert audio to mono
    audio = audio.set_channels(1)
    
    # Normalize the audio
    normalized_audio = audio.apply_gain(-audio.max_dBFS)
    
    # Find silent parts
    silent_ranges = []
    for i in range(0, len(audio), min_silence_len):
        chunk = audio[i:i + min_silence_len]
        if chunk.dBFS < threshold:
            silent_ranges.append(i)
    
    # Use scipy to find peaks in silent parts
    peaks, _ = find_peaks(np.diff(silent_ranges), height=1)

    # Save each slice
    start = 0
    slice_count = start_slice_index  # Initialize slice count from where it left off
    for i, peak in enumerate(peaks):
        end = silent_ranges[peak]
        slice = audio[start:end]
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_{slice_count}.wav")
        slice.export(output_file, format="wav")
        start = end
        slice_count += 1  # Increment slice count for the next slice
    
    return slice_count  # Return the updated slice count

# Usage example
input_dir = "/Users/nathan/Documents/pyproj/sample_slice/kick"
output_dir = "/Users/nathan/Documents/pyproj/sample_slice/output"
split_audio_directory(input_dir, output_dir)
